// Copyright 2021 Datafuse Labs.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use common_arrow::arrow_format::flight::service::flight_service_server::FlightServiceServer;
use common_base::base::tokio;
use common_base::base::tokio::net::TcpListener;
use common_base::base::tokio::sync::Notify;
use common_config::Setting;
use common_exception::ErrorCode;
use common_exception::Result;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Identity;
use tonic::transport::Server;
use tonic::transport::ServerTlsConfig;
use tracing::info;

use crate::api::rpc::DatabendQueryFlightService;
use crate::servers::Server as DatabendQueryServer;

pub struct RpcService {
    pub config: Setting,
    pub abort_notify: Arc<Notify>,
}

impl RpcService {
    pub fn create(config: Setting) -> Result<Box<dyn DatabendQueryServer>> {
        Ok(Box::new(Self {
            config,
            abort_notify: Arc::new(Notify::new()),
        }))
    }

    async fn listener_tcp(listening: SocketAddr) -> Result<(TcpListenerStream, SocketAddr)> {
        let listener = TcpListener::bind(listening).await.map_err(|e| {
            ErrorCode::TokioError(format!("{{{}:{}}} {}", listening.ip(), listening.port(), e))
        })?;
        let listener_addr = listener.local_addr()?;
        Ok((TcpListenerStream::new(listener), listener_addr))
    }

    fn shutdown_notify(&self) -> impl Future<Output = ()> + 'static {
        let notified = self.abort_notify.clone();
        async move {
            notified.notified().await;
        }
    }

    async fn server_tls_config(conf: &Setting) -> Result<ServerTlsConfig> {
        let cert = tokio::fs::read(conf.query.rpc_tls_server_cert.as_str()).await?;
        let key = tokio::fs::read(conf.query.rpc_tls_server_key.as_str()).await?;
        let server_identity = Identity::from_pem(cert, key);
        let tls_conf = ServerTlsConfig::new().identity(server_identity);
        Ok(tls_conf)
    }

    pub async fn start_with_incoming(&mut self, listener_stream: TcpListenerStream) -> Result<()> {
        let flight_api_service = DatabendQueryFlightService::create();
        let builder = Server::builder();
        let mut builder = if self.config.tls_rpc_server_enabled() {
            info!("databend query tls rpc enabled");
            builder
                .tls_config(Self::server_tls_config(&self.config).await.map_err(|e| {
                    ErrorCode::TLSConfigurationFailure(format!(
                        "failed to load server tls config: {e}",
                    ))
                })?)
                .map_err(|e| {
                    ErrorCode::TLSConfigurationFailure(format!("failed to invoke tls_config: {e}",))
                })?
        } else {
            builder
        };

        let server = builder
            .add_service(FlightServiceServer::new(flight_api_service))
            .serve_with_incoming_shutdown(listener_stream, self.shutdown_notify());

        tokio::spawn(server);
        Ok(())
    }
}

#[async_trait::async_trait]
impl DatabendQueryServer for RpcService {
    async fn shutdown(&mut self, _graceful: bool) {}

    async fn start(&mut self, listening: SocketAddr) -> Result<SocketAddr> {
        let (listener_stream, listener_addr) = Self::listener_tcp(listening).await?;
        self.start_with_incoming(listener_stream).await?;
        Ok(listener_addr)
    }
}
