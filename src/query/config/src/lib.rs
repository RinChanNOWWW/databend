// Copyright 2022 Datafuse Labs.
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

#![allow(clippy::uninlined_format_args)]
#![feature(no_sanitize)]

mod config;
/// Config mods provide config support.
///
/// We are providing two config types:
///
/// - [`config::Config`] represents the options from command line , configuration files or environment vars.
/// - [`setting::Setting`] "internal representation" of application settings .
/// - [`global::GlobalSetting`] A global singleton of [`crate::Setting`].
///
/// It's safe to refactor [`setting::Setting`] in anyway, as long as it satisfied the following traits
///
/// - `TryInto<setting::Setting> for config::Config`
/// - `From<setting::Setting> for config::Config`
mod global;
mod setting;
mod version;

pub use config::Config;
pub use config::ExternalCacheStorageTypeConfig;
pub use config::QueryConfig;
pub use config::StorageConfig;
pub use global::GlobalSetting;
pub use setting::CacheSetting;
pub use setting::CatalogHiveSetting;
pub use setting::CatalogSetting;
pub use setting::ExternalCacheStorageTypeSetting;
pub use setting::QuerySetting;
pub use setting::Setting;
pub use setting::ThriftProtocol;
pub use version::DATABEND_COMMIT_VERSION;
pub use version::QUERY_SEMVER;
