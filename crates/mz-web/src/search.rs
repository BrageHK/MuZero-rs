//! Single-tree Gumbel MuZero search, async so it works on WebGPU (which has no
//! synchronous readback in the browser). A port of the Gumbel path of
//! `mz-train`'s `batched_search` with the batch dimension and PUCT dropped.

use burn::{
    Tensor,
    tensor::{Int, TensorData, Transaction, backend::Backend},
};
use mz_core::networks::MuZeroNets;
use mz_core::support::logits_to_scalars;

use crate::rng::Rng;

const RESCALE_EPS: f32 = 1e-8;

#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub num_simulations: usize,
    pub action_space: usize,
    pub support_size: usize,
    pub discount: f32,
    pub max_num_considered_actions: usize,
    pub c_visit: f32,
    pub c_scale: f32,
    pub two_player: bool,
}

pub struct SearchResult {
    pub best_action: usize,
    /// v_mix at the root, from the perspective of the side to move.
    pub value: f32,
    /// The improved policy, i.e. the training target.
    pub policy: Vec<f32>,
}

struct Node {
    visits: usize,
    action: usize,
    hidden: usize,
    first_child: usize,
    cumulative_value: f32,
    reward: f32,
    prior: f32,
    logit: f32,
    value: f32,
    legal: bool,
}

pub async fn gumbel_search<B: Backend, N: MuZeroNets<B>>(
    net: &N,
    device: &B::Device,
    obs: &[f32],
    legal: &[bool],
    params: &SearchParams,
    num_simulations: usize,
    rng: Option<Rng>,
) -> SearchResult {
    let action_space = params.action_space;
    let discount = params.discount;
    let value_sign = if params.two_player { -1.0f32 } else { 1.0f32 };
    let support_size = params.support_size;

    let obs = Tensor::<B, 1>::from_floats(obs, device).reshape([1, obs.len()]);
    let (root_hidden, _, root_value, root_policy) = net.initial_inference(obs);

    let [root_value, root_policy] = Transaction::default()
        .register(root_value)
        .register(root_policy)
        .execute_async()
        .await
        .expect("root inference readback")
        .try_into()
        .expect("Correct amount of tensor data");
    let root_value = logits_to_scalars(&root_value.into_vec::<f32>().unwrap(), 1, support_size)[0];
    let logits = root_policy.into_vec::<f32>().unwrap();

    let legal_actions: Vec<usize> = (0..action_space).filter(|&a| legal[a]).collect();
    assert!(!legal_actions.is_empty(), "no legal action to search");
    if legal_actions.len() == 1 {
        let action = legal_actions[0];
        let mut policy = vec![0.0f32; action_space];
        policy[action] = 1.0;
        return SearchResult {
            best_action: action,
            value: root_value,
            policy,
        };
    }

    let gumbel: Vec<f32> = match rng {
        // The Gumbel perturbation is the algorithm's exploration; zeroing it
        // makes the move deterministic.
        Some(mut rng) => (0..action_space).map(|_| rng.gumbel()).collect(),
        None => vec![0.0; action_space],
    };
    let num_considered = legal_actions
        .len()
        .min(params.max_num_considered_actions.min(action_space));
    let considered = sequence_of_considered_visits(num_considered, num_simulations);

    let mut hiddens = vec![root_hidden];
    let mut nodes: Vec<Node> = Vec::with_capacity(1 + (num_simulations + 1) * action_space);
    nodes.push(Node {
        visits: 0,
        action: 0,
        hidden: 0,
        first_child: 1,
        cumulative_value: 0.0,
        reward: 0.0,
        prior: 0.0,
        logit: 0.0,
        value: root_value,
        legal: true,
    });

    let priors = masked_softmax(&logits, Some(legal));
    for (action, &prior) in priors.iter().enumerate() {
        nodes.push(Node {
            visits: 0,
            action,
            hidden: 0,
            first_child: 0,
            cumulative_value: 0.0,
            reward: 0.0,
            prior,
            logit: if legal[action] {
                logits[action]
            } else {
                f32::NEG_INFINITY
            },
            value: 0.0,
            legal: legal[action],
        });
    }

    let mut scores: Vec<f32> = Vec::with_capacity(action_space);
    let mut path: Vec<usize> = Vec::new();

    for sim_step in 0..num_simulations {
        let mut node_idx = 0;
        path.clear();
        path.push(0);
        while nodes[node_idx].first_child != 0 {
            let (sum_visits, _) = completed_q_scaled(
                &nodes,
                node_idx,
                action_space,
                discount,
                value_sign,
                params.c_visit,
                params.c_scale,
                &mut scores,
            );
            node_idx = if node_idx == 0 {
                gumbel_root_select(&nodes, action_space, &gumbel, &scores, considered[sim_step])
            } else {
                improved_policy_select(&nodes, node_idx, action_space, &mut scores, sum_visits)
            };
            path.push(node_idx);
        }

        let [.., parent_idx, leaf_idx] = path.as_slice() else {
            unreachable!()
        };
        let (parent_idx, leaf_idx) = (*parent_idx, *leaf_idx);

        let hidden = hiddens[nodes[parent_idx].hidden].clone();
        let action = Tensor::<B, 1, Int>::from_data(
            TensorData::from([nodes[leaf_idx].action as i64].as_slice()),
            device,
        );
        let (new_hidden, reward, value, policy) = net.recurrent_inference(hidden, action, action_space);

        let [reward, value, policy] = Transaction::default()
            .register(reward)
            .register(value)
            .register(policy)
            .execute_async()
            .await
            .expect("recurrent inference readback")
            .try_into()
            .expect("Correct amount of tensor data");
        let reward = logits_to_scalars(&reward.into_vec::<f32>().unwrap(), 1, support_size)[0];
        let value = logits_to_scalars(&value.into_vec::<f32>().unwrap(), 1, support_size)[0];
        let logits = policy.into_vec::<f32>().unwrap();

        let first_child = nodes.len();
        let priors = masked_softmax(&logits, None);
        for (action, &prior) in priors.iter().enumerate() {
            nodes.push(Node {
                visits: 0,
                action,
                hidden: 0,
                first_child: 0,
                cumulative_value: 0.0,
                reward: 0.0,
                prior,
                logit: logits[action],
                value: 0.0,
                // Legality is unknown in latent space, so only the root masks.
                legal: true,
            });
        }

        hiddens.push(new_hidden);
        let leaf = &mut nodes[leaf_idx];
        leaf.hidden = hiddens.len() - 1;
        leaf.reward = reward;
        leaf.value = value;
        leaf.first_child = first_child;

        let mut back_value = value;
        for &idx in path.iter().rev() {
            let node = &mut nodes[idx];
            node.visits += 1;
            node.cumulative_value += back_value;
            back_value = node.reward + discount * value_sign * back_value;
        }
    }

    let (_, v_mix) = completed_q_scaled(
        &nodes,
        0,
        action_space,
        discount,
        value_sign,
        params.c_visit,
        params.c_scale,
        &mut scores,
    );

    let first_child = nodes[0].first_child;
    let children = &nodes[first_child..first_child + action_space];

    let mut policy: Vec<f32> = children
        .iter()
        .map(|child| {
            if child.legal {
                child.logit + scores[child.action]
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    softmax_in_place(&mut policy);

    let max_visits = children
        .iter()
        .filter(|child| child.legal)
        .map(|child| child.visits)
        .max()
        .unwrap_or(0);

    let mut best_action = 0;
    let mut best_score = f32::NEG_INFINITY;
    for child in children {
        if !child.legal || child.visits != max_visits {
            continue;
        }
        let score = gumbel[child.action] + child.logit + scores[child.action];
        if score > best_score {
            best_score = score;
            best_action = child.action;
        }
    }

    SearchResult {
        best_action,
        value: v_mix,
        policy,
    }
}

/// Completed Q values, min-max rescaled by the sigma transform, plus v_mix.
fn completed_q_scaled(
    nodes: &[Node],
    node_idx: usize,
    action_space: usize,
    discount: f32,
    value_sign: f32,
    c_visit: f32,
    c_scale: f32,
    out: &mut Vec<f32>,
) -> (usize, f32) {
    let first_child = nodes[node_idx].first_child;
    let children = &nodes[first_child..first_child + action_space];

    let child_q = |child: &Node| {
        child.reward + discount * value_sign * child.cumulative_value / child.visits as f32
    };

    let mut sum_visits = 0usize;
    let mut max_visits = 0usize;
    let mut sum_priors = 0.0f32;
    let mut weighted_q = 0.0f32;
    for child in children {
        if child.visits == 0 {
            continue;
        }
        sum_visits += child.visits;
        max_visits = max_visits.max(child.visits);
        sum_priors += child.prior;
        weighted_q += child.prior * child_q(child);
    }

    let raw_value = nodes[node_idx].value;
    let v_mix = if sum_visits == 0 || sum_priors <= 0.0 {
        raw_value
    } else {
        (raw_value + sum_visits as f32 * weighted_q / sum_priors) / (sum_visits as f32 + 1.0)
    };

    out.clear();
    let mut min_q = f32::INFINITY;
    let mut max_q = f32::NEG_INFINITY;
    for child in children {
        let q = if child.visits == 0 {
            v_mix
        } else {
            child_q(child)
        };
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        out.push(q);
    }

    let visit_scale = (c_visit + max_visits as f32) * c_scale;
    let range = max_q - min_q;
    for q in out.iter_mut() {
        *q = visit_scale * (*q - min_q) / (range + RESCALE_EPS);
    }

    (sum_visits, v_mix)
}

/// Sequential Halving: among the legal children sitting at the current visit
/// threshold, take the best Gumbel-perturbed score.
fn gumbel_root_select(
    nodes: &[Node],
    action_space: usize,
    gumbel: &[f32],
    scores: &[f32],
    threshold: usize,
) -> usize {
    let first_child = nodes[0].first_child;
    let mut best_node = usize::MAX;
    let mut best_score = f32::NEG_INFINITY;
    let mut fallback = first_child;
    let mut fewest_visits = usize::MAX;

    for (child_idx, child) in nodes.iter().enumerate().skip(first_child).take(action_space) {
        if !child.legal {
            continue;
        }
        if child.visits < fewest_visits {
            fewest_visits = child.visits;
            fallback = child_idx;
        }
        if child.visits != threshold {
            continue;
        }
        let score = gumbel[child.action] + child.logit + scores[child.action];
        if score > best_score {
            best_score = score;
            best_node = child_idx;
        }
    }

    if best_node == usize::MAX {
        fallback
    } else {
        best_node
    }
}

fn improved_policy_select(
    nodes: &[Node],
    node_idx: usize,
    action_space: usize,
    scores: &mut [f32],
    sum_visits: usize,
) -> usize {
    let first_child = nodes[node_idx].first_child;
    let children = &nodes[first_child..first_child + action_space];

    for child in children {
        scores[child.action] = if child.legal {
            child.logit + scores[child.action]
        } else {
            f32::NEG_INFINITY
        };
    }
    softmax_in_place(scores);

    let denom = 1.0 + sum_visits as f32;
    let mut best_node = first_child;
    let mut best = f32::NEG_INFINITY;
    for (child_idx, child) in children.iter().enumerate() {
        if !child.legal {
            continue;
        }
        let score = scores[child.action] - child.visits as f32 / denom;
        if score > best {
            best = score;
            best_node = first_child + child_idx;
        }
    }
    best_node
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in values.iter_mut() {
        *v /= sum;
    }
}

fn masked_softmax(logits: &[f32], mask: Option<&[bool]>) -> Vec<f32> {
    let mut out: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(action, &logit)| {
            if mask.is_none_or(|m| m[action]) {
                logit
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    softmax_in_place(&mut out);
    out
}

fn sequence_of_considered_visits(num_considered: usize, num_simulations: usize) -> Vec<usize> {
    if num_considered <= 1 {
        return (0..num_simulations).collect();
    }

    let log2max = usize::BITS as usize - (num_considered - 1).leading_zeros() as usize;
    let mut visits = vec![0usize; num_considered];
    let mut considered = num_considered;
    let mut sequence = Vec::with_capacity(num_simulations);

    while sequence.len() < num_simulations {
        let extra_visits = (num_simulations / (log2max * considered)).max(1);
        for _ in 0..extra_visits {
            sequence.extend_from_slice(&visits[..considered]);
            for visit in visits[..considered].iter_mut() {
                *visit += 1;
            }
        }
        considered = (considered / 2).max(2);
    }

    sequence.truncate(num_simulations);
    sequence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn considered_visits_match_mctx() {
        assert_eq!(
            sequence_of_considered_visits(4, 8),
            vec![0, 0, 0, 0, 1, 1, 2, 2]
        );
        assert_eq!(sequence_of_considered_visits(2, 4), vec![0, 0, 1, 1]);
        assert_eq!(sequence_of_considered_visits(1, 5), vec![0, 1, 2, 3, 4]);
    }
}
