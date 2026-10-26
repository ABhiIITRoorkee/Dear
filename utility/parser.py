# utility/parser.py

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run HCF.")

    # ---------------- General / IO ----------------
    parser.add_argument('--weights_path', nargs='?', default='')
    parser.add_argument('--output_path', nargs='?', default='')
    parser.add_argument('--method', type=str, default='kd', choices=['baseline', 'kd'])

    # ---------------- Hypergraph thresholds ----------------
    parser.add_argument('--alpha1', type=float, default=0.6)
    parser.add_argument('--alpha2', type=float, default=0.6)
    parser.add_argument('--alpha3', type=float, default=0.8)
    parser.add_argument('--alpha4', type=float, default=0.8)

    # ---------------- Retrieval / topK ----------------
    parser.add_argument('--similar_threthold', type=float, default=5.0)

    # ---------------- Data / runtime ----------------
    parser.add_argument('--data_path', nargs='?', default='dataset/')
    parser.add_argument('--proj_path', nargs='?', default=r'')
    parser.add_argument('--save_recom', type=int, default=1)
    parser.add_argument('--dataset', nargs='?', default='fold_1')
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=500)

    # ---------------- Model / optimizer ----------------
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--regs', nargs='?', default='[1e-4]')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--model_type', nargs='?', default='hcf')
    parser.add_argument('--adj_type', nargs='?', default='norm_adj')
    parser.add_argument('--alg_type', nargs='?', default='hcf')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]')
    parser.add_argument('--drop_edge', type=float, default=0.95)

    # ---------------- Evaluation ----------------
    parser.add_argument('--Ks', nargs='?', default='[5, 20]')
    parser.add_argument('--save_flag', type=int, default=1)
    parser.add_argument('--test_flag', nargs='?', default='part')
    parser.add_argument('--report', type=int, default=0)

    # ---------------- Logging extras ----------------
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--print_embed_every', type=int, default=1)
    parser.add_argument('--print_hgraph_every', type=int, default=0)

    # ---------------- KD / Router Distillation ----------------
    parser.add_argument('--use_kd_router', type=int, default=1)

    # BEST FOUND SO FAR:
    # local_peak_topM500_mix0.7_l0.047_tau0.25_topk200_pool5_every2_sim5
    parser.add_argument('--kd_lambda', type=float, default=0.042)
    parser.add_argument('--kd_tau', type=float, default=0.25)
    parser.add_argument('--kd_topk', type=int, default=200)
    parser.add_argument('--kd_every', type=int, default=5)
    parser.add_argument('--router_hidden', type=int, default=256)

    parser.add_argument(
        '--kd_pool_mult', type=int, default=5,
        help='Pool size multiplier for KD candidates: pool_size = kd_pool_mult * kd_topk.'
    )

    parser.add_argument(
        '--kd_warmup_epochs', type=int, default=20,
        help='Epochs to train without KD at the start.'
    )
    parser.add_argument(
        '--kd_ramp_epochs', type=int, default=80,
        help='If >0, linearly ramps KD weight after warmup over kd_ramp_epochs.'
    )

    parser.add_argument(
        '--kd_teacher_mix', type=float, default=0.7,
        help='Teacher mix in [0,1]. 1.0 = pure MPNet cosine; 0.0 = pure usage-jaccard.'
    )

    # NEW: inference rerank retrieval size
    parser.add_argument(
        '--rerank_topM', type=int, default=500,
        help='If >0, retrieve topM by cosine then router-rerank. If 0, uses kd_topk*kd_pool_mult.'
    )

    # NEW: missing-tag handling
    parser.add_argument(
        '--skip_tag_loss_when_missing', type=int, default=0, choices=[0, 1],
        help='If 1, tag losses are skipped when positive tags are missing (Data returns -1 for missing).'
    )

    # ---------------- Optional: Geometry KD ----------------
    parser.add_argument('--use_geom_kd', type=int, default=0)
    parser.add_argument('--geom_lambda', type=float, default=0.05)
    parser.add_argument('--geom_batch', type=int, default=256)

    # ============================================================
    # NEW: Latent Hypergraph Inference (SPHINX-style)
    # ============================================================
    parser.add_argument(
        '--use_latent_hypergraph', type=int, default=0, choices=[0, 1],
        help='If 1, replace selected heuristic hypergraphs with learned (latent) hypergraphs.'
    )

    parser.add_argument(
        '--latent_targets', type=str, default='user',
        choices=['user', 'item', 'mtag', 'atag', 'all'],
        help='Which hypergraph(s) are learned when use_latent_hypergraph=1.'
    )

    parser.add_argument(
        '--hin_num_edges', type=int, default=256,
        help='Number of latent hyperedges M (incidence matrix H is N x M).'
    )

    parser.add_argument(
        '--hin_topk_per_node', type=int, default=4,
        help='Each node connects to top-k latent hyperedges (discrete sparsification).'
    )
    parser.add_argument(
        '--hin_tau', type=float, default=0.5,
        help='Temperature for softmax / gumbel-softmax in latent incidence.'
    )
    parser.add_argument(
        '--hin_straight_through', type=int, default=1, choices=[0, 1],
        help='If 1, use straight-through hard topk (discrete) with soft gradients.'
    )

    parser.add_argument(
        '--hin_hidden', type=int, default=256,
        help='Hidden size for incidence predictor MLP.'
    )
    parser.add_argument(
        '--hin_layers', type=int, default=2,
        help='Number of MLP layers in incidence predictor (>=1).'
    )
    parser.add_argument(
        '--hin_dropout', type=float, default=0.1,
        help='Dropout inside incidence predictor.'
    )

    parser.add_argument(
        '--hin_refresh_every', type=int, default=1,
        help='Recompute latent hypergraph every N epochs. 0 means compute once and freeze.'
    )
    parser.add_argument(
        '--hin_warmup_epochs', type=int, default=0,
        help='Epochs to train with heuristic hypergraphs before enabling latent inference.'
    )

    parser.add_argument(
        '--hin_reg_entropy', type=float, default=0.0,
        help='Entropy regularizer weight on node->edge assignment distribution.'
    )
    parser.add_argument(
        '--hin_reg_balance', type=float, default=0.0,
        help='Encourage balanced use of latent hyperedges (prevents all nodes picking same edges).'
    )
    parser.add_argument(
        '--hin_reg_l2', type=float, default=0.0,
        help='L2 regularization weight on predictor parameters (separate from regs).'
    )

    parser.add_argument(
        '--kd_teacher_from_latent', type=int, default=0, choices=[0, 1],
        help='If 1 and use_kd_router=1, build KD teacher distribution from latent hypergraph instead of cosine/Jaccard.'
    )

    # ---------------- Run control (run.py) ----------------
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--fold_start', type=int, default=1)
    parser.add_argument('--fold_end', type=int, default=10)

    parser.add_argument('--run_tag', type=str, default='')

    parser.add_argument(
        '--infer_tau', type=float, default=1.0,
        help='temperature for router neighbor weighting during inference'
    )

    return parser.parse_args()
