//! Runtime-selected optimizer behind a single type, so training loops are
//! compiled once instead of once per optimizer.

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{
    Adam, AdamConfig, AdamW, AdamWConfig, GradientsParams, LearningRate, MultiGradientsParams,
    Optimizer, Sgd, SgdConfig,
};
use burn::record::{PrecisionSettings, Record};
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use crate::mz_config::{MuZeroConfig, OptimChoice};

// State/serialized-state of one wrapped optimizer, named via the adaptor's own
// associated types so we never spell out burn's internal container types.
type AdaptorState<O, M, B> = <OptimizerAdaptor<O, M, B> as Optimizer<M, B>>::Record;
type AdaptorStateItem<O, M, B, S> = <AdaptorState<O, M, B> as Record<B>>::Item<S>;

/// Wraps the concrete burn optimizers picked by `optimizer:` in the config.
#[derive(Clone)]
pub enum AnyOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    Adam(OptimizerAdaptor<Adam, M, B>),
    AdamW(OptimizerAdaptor<AdamW, M, B>),
    Sgd(OptimizerAdaptor<Sgd<B::InnerBackend>, M, B>),
}

impl<B, M> AnyOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    pub fn new(mz_conf: &MuZeroConfig) -> Self {
        match mz_conf.optimizer {
            OptimChoice::Adam => Self::Adam(AdamConfig::new().init()),
            OptimChoice::AdamW => Self::AdamW(AdamWConfig::new().init()),
            OptimChoice::Sgd => Self::Sgd(SgdConfig::new().init()),
        }
    }
}

pub enum AnyOptimizerRecord<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    Adam(AdaptorState<Adam, M, B>),
    AdamW(AdaptorState<AdamW, M, B>),
    Sgd(AdaptorState<Sgd<B::InnerBackend>, M, B>),
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub enum AnyOptimizerRecordItem<B, M, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    S: PrecisionSettings,
{
    Adam(AdaptorStateItem<Adam, M, B, S>),
    AdamW(AdaptorStateItem<AdamW, M, B, S>),
    Sgd(AdaptorStateItem<Sgd<B::InnerBackend>, M, B, S>),
}

impl<B, M> Record<B> for AnyOptimizerRecord<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    type Item<S: PrecisionSettings> = AnyOptimizerRecordItem<B, M, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            Self::Adam(r) => AnyOptimizerRecordItem::Adam(r.into_item()),
            Self::AdamW(r) => AnyOptimizerRecordItem::AdamW(r.into_item()),
            Self::Sgd(r) => AnyOptimizerRecordItem::Sgd(r.into_item()),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        match item {
            AnyOptimizerRecordItem::Adam(i) => Self::Adam(Record::from_item(i, device)),
            AnyOptimizerRecordItem::AdamW(i) => Self::AdamW(Record::from_item(i, device)),
            AnyOptimizerRecordItem::Sgd(i) => Self::Sgd(Record::from_item(i, device)),
        }
    }
}

impl<B, M> Optimizer<M, B> for AnyOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    type Record = AnyOptimizerRecord<B, M>;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        match self {
            Self::Adam(o) => o.step(lr, module, grads),
            Self::AdamW(o) => o.step(lr, module, grads),
            Self::Sgd(o) => o.step(lr, module, grads),
        }
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        match self {
            Self::Adam(o) => o.step_multi(lr, module, grads),
            Self::AdamW(o) => o.step_multi(lr, module, grads),
            Self::Sgd(o) => o.step_multi(lr, module, grads),
        }
    }

    fn to_record(&self) -> Self::Record {
        match self {
            Self::Adam(o) => AnyOptimizerRecord::Adam(o.to_record()),
            Self::AdamW(o) => AnyOptimizerRecord::AdamW(o.to_record()),
            Self::Sgd(o) => AnyOptimizerRecord::Sgd(o.to_record()),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        match (self, record) {
            (Self::Adam(o), AnyOptimizerRecord::Adam(r)) => Self::Adam(o.load_record(r)),
            (Self::AdamW(o), AnyOptimizerRecord::AdamW(r)) => Self::AdamW(o.load_record(r)),
            (Self::Sgd(o), AnyOptimizerRecord::Sgd(r)) => Self::Sgd(o.load_record(r)),
            _ => panic!(
                "optimizer record does not match the `optimizer:` currently set in the config"
            ),
        }
    }
}
