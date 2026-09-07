import argparse
 
 
def load_config():
    parser = argparse.ArgumentParser(
        description="TFUS Neural Operator — Training and Evaluation Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    g = parser.add_argument_group("general")
    g.add_argument('--data_path',     type=str,   default='./',
                   help='Root path containing .h5 data')
    g.add_argument('--run_name',      type=str,   default='exp',
                   help='Name of the current run; used for output dir / wandb')
    g.add_argument('--output_dir',    type=str,   default='./runs',
                   help='Root directory for checkpoints/logs (subdir = run_name)')
    g.add_argument('--seed',          type=int,   default=77,
                   help='Global random seed (numpy / torch / cuda)')
    g.add_argument('--gpu_num',       type=int,   default=0,
                   help='GPU index. Use -1 for CPU (debug only)')
 
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    d = parser.add_argument_group("data")
    d.add_argument('--data_shape',    type=int,   default=112,
                   help='ROI side length in voxels (cube assumed)')
    d.add_argument('--downsample',    type=int,   default=1,
                   help='Downsampling rate for input data')               
    d.add_argument('--frequencies',   type=int,   nargs='+',
                   default=[250000, 400000, 500000],
                   help='Frequencies (Hz) to include in train/val/test')
    d.add_argument('--skull_modality', type=str,  default='CT',
                   choices=['CT', 'MR'],
                   help='Which skull volume modality to use as input')
    d.add_argument('--patch_vox',     type=int,   default=4,
                   help='Patch side length in voxels (encoder Conv3d stride)')
    d.add_argument('--domain_size_vox', type=int, nargs=3,
                   default=[450, 450, 300],
                   help='Full simulation domain size in voxels (Nx Ny Nz)')
    d.add_argument('--voxel_mm',      type=float, default=0.5,
                   help='Isotropic voxel spacing in mm')
    d.add_argument('--normalize_field', action='store_true', default=False,
                   help='Apply sign-log compression to ff/pmax in dataset')
    d.add_argument('--mirror_aug', action='store_true', default=False,
                   help='Enable sagittal mirror augmentation on the TRAIN split')
    d.add_argument('--mirror_prob', type=float, default=0.5,
                   help='Per-sample probability of applying the mirror flip')
    d.add_argument('--mirror_axis', type=int, default=0, choices=[0, 1],
                   help='Axis to flip during augmentation')
    d.add_argument('--num_workers',   type=int,   default=4,
                   help='DataLoader workers per loader')
    d.add_argument('--split_train',   type=str,   nargs='+', default=None,
                   help='Override training skull IDs, e.g. S01 S02 ...')
    d.add_argument('--split_val',     type=str,   nargs='+', default=None,
                   help='Override validation skull IDs')
    d.add_argument('--split_test',    type=str,   nargs='+', default=None,
                   help='Override test skull IDs')
    d.add_argument('--positions',     type=int,   nargs='+', default=None,
                   help='Subset of position indices [0, 300). None -> all')
    d.add_argument('--eval_heldout_per_pair', type=int, default=30,
                   help='Reserve this many POSITIONS per train-skull as a held-out '
                        'eval set (seen skulls, unseen positions), excluded from '
                        'training across ALL frequencies. 0 disables. '
                        'e.g. 30 of 300 = 10%%.')
 
    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    m = parser.add_argument_group("model")
    m.add_argument('--embed_dim',     type=int,   default=384,
                   help='Hidden dim d (must be divisible by 6 and num_heads)')
    m.add_argument('--num_latents',   type=int,   default=512,
                   help='M, number of latent tokens after Perceiver pool')
    m.add_argument('--depth',         type=int,   default=8,
                   help='L, number of latent processor transformer blocks')
    m.add_argument('--num_heads',     type=int,   default=6,
                   help='Attention heads')
    m.add_argument('--mlp_ratio',     type=float, default=4.0,
                   help='MLP expansion ratio in all blocks')
    m.add_argument('--dropout',       type=float, default=0.0,
                   help='Attention / MLP dropout')
    m.add_argument('--no_dit',        action='store_true', default=False,
                   help='Disable DiT conditioning entirely (unconditional baseline)')
    m.add_argument('--coord_max_freq', type=float, default=200,
                   help='Cap the TOP angular frequency of the coordinate Fourier PE '
                        '(all coord PEs: encoder keys, decoder query, skull-read, '
                        'intra-patch).')
    m.add_argument('--ct_stem_depth', type=int, default=0,
                   help='Deep local CT stem: # stride-1 ResBlocks before patch '
                        'tokenize (0 = plain PatchEmbed3D). RF = 3 + 4*depth voxels. '
                        'Only the skull pathway; field unchanged.')
    m.add_argument('--ct_stem_channels', type=int, default=64,
                   help='Channel width of the deep CT stem (when ct_stem_depth>0)')

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    c = parser.add_argument_group("conditioning")
    c.add_argument('--freq_range_hz', type=float, nargs=2,
                   default=[200e3, 550e3],
                   help='Min/max frequency for normalization (Hz)')
    c.add_argument('--cond_dropout',  type=float, default=0.1,
                   help='Per-source null dropout (CFG-style)')
    c.add_argument('--cond_sources', type=str, nargs='+',
                default=['freq', 'pos', 'angle'],
                choices=['freq', 'pos', 'angle'],
                help='Which conditioning sources to use. Subset of {freq, pos, angle}. '
                        'Embedders are created only for declared sources.')
 
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    t = parser.add_argument_group("training")
    t.add_argument('--criterion',     type=str,   default='mse',
                   choices=['mse', 'l1', 'huber', 'log_mse'],
                   help='Reconstruction (recon) base loss')
    t.add_argument('--loss_terms', type=str, nargs='+', default=['recon'],
                   choices=['recon', 'dice', 'peak_dist', 'peak_diff'],
                   help='Loss terms to combine. recon = voxelwise --criterion; '
                        'dice/peak_dist/peak_diff are soft differentiable analogs '
                        'of the eval metrics.')
    t.add_argument('--loss_weighting', type=str, default=None,
                   choices=['uncertainty', 'fixed'],
                   help='Multi-term weighting. uncertainty = learnable (Kendall, '
                        'collapse-safe); fixed = constant --loss_fixed_weights. '
                        'Default: fixed if only recon, else uncertainty.')
    t.add_argument('--loss_fixed_weights', type=float, nargs='+', default=None,
                   help='Per-term weights (fixed mode), same order/length as --loss_terms')
    t.add_argument('--peak_beta', type=float, default=30.0,
                   help='Sharpness for soft-argmax / soft-max (peak_dist, peak_diff, '
                        'and the dice threshold reference)')
    t.add_argument('--mask_beta', type=float, default=10.0,
                   help='Sharpness of the soft FWHM membership sigmoid (dice)')
    t.add_argument('--batch_size',    type=int,   default=4,
                   help='Per-step batch size (train/val/test)')
    t.add_argument('--learning_rate', type=float, default=2e-4,
                   help='Peak learning rate (post-warmup)')
    t.add_argument('--weight_decay',  type=float, default=0.05,
                   help='AdamW weight decay (excluding bias/norm params)')
    t.add_argument('--num_epochs',    type=int,   default=200,
                   help='Maximum number of training epochs')
    t.add_argument('--warmup_epochs', type=int,   default=10,
                   help='Linear warmup epochs before cosine decay')
    t.add_argument('--lr_schedule',   action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Cosine LR schedule with linear warmup. '
                        'Use --no-lr_schedule for constant LR.')
    t.add_argument('--patience',      type=int,   default=50,
                   help='Early-stopping patience (epochs without val improvement)')
    t.add_argument('--grad_clip',     type=float, default=1.0,
                   help='Max global grad norm; <=0 disables clipping')
    t.add_argument('--precision',     type=str,   default='bf16',
                   choices=['fp32', 'bf16', 'fp16'],
                   help='Mixed-precision dtype for forward/backward')
    t.add_argument('--best_metric',   type=str,   default='dice',
               choices=['loss', 'dice'],
               help='Metric tracked for best-ckpt & early stopping. '
                    'loss -> min mode (val_loss); dice -> max mode (val dice).')
 
    # ------------------------------------------------------------------
    # Checkpointing / logging
    # ------------------------------------------------------------------
    l = parser.add_argument_group("logging")
    l.add_argument('--save_every',    type=int,   default=20,
                   help='Save checkpoint every N epochs (best ckpt always saved)')
    l.add_argument('--log_every',     type=int,   default=50,
                   help='Print/log training stats every N steps')
    l.add_argument('--wandb_pj',      type=str,   default=None,
                   help='WandB project name. None disables WandB logging')
    l.add_argument('--resume',        type=str,   default=None,
                   help='Path to checkpoint .pt to resume from')
 
    # ------------------------------------------------------------------
    # Plotting / evaluation
    # ------------------------------------------------------------------
    p = parser.add_argument_group("plotting")
    p.add_argument('--plot',                    action='store_true', default=False,
                    help='Save predicted vs target field slices during eval')
    p.add_argument('--collect_predictions',     action='store_true', default=False,
                    help='Collect predicted results to cpu')
 
    return parser.parse_args()