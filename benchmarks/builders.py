from torch.utils.data import DataLoader, Subset


def build_mpnn(args, device):
    # Import inside function to avoid heavy imports at module-import time
    from mpnn import GraphTaskDataset, GIN, GraphMPNNTrainer, collate_fn

    train_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "train")
    valid_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "valid")
    test_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "test")

    # Optionally limit dataset sizes via args (set in model config or CLI overrides)
    try:
        if getattr(args, 'max_samples_train', None) and args.max_samples_train > 0:
            n = min(len(train_dataset), int(args.max_samples_train))
            train_dataset = Subset(train_dataset, list(range(n)))
        if getattr(args, 'max_samples_valid', None) and args.max_samples_valid > 0:
            n = min(len(valid_dataset), int(args.max_samples_valid))
            valid_dataset = Subset(valid_dataset, list(range(n)))
        if getattr(args, 'max_samples_test', None) and args.max_samples_test > 0:
            n = min(len(test_dataset), int(args.max_samples_test))
            test_dataset = Subset(test_dataset, list(range(n)))
    except Exception:
        # Keep original datasets on any error
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GIN(in_features=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, out_features=1, dropout=0.5)
    task_type = "classification" if args.task in ["cycle_check"] else "regression"
    trainer = GraphMPNNTrainer(
        model,
        learning_rate=args.learning_rate,
        device=device,
        task_type=task_type,
        loss=getattr(args, 'loss', None),
    )

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": task_type,
    }


def build_transformer(args, device, which):
    # Lazy import to avoid heavy module loads during import-time checks
    if which == "graph_transformer":
        from graph_transformer import GraphDataset, GraphTransformer as Transformer
    else:
        from autograph_transformer import GraphDataset as AGDataset, GraphTransformer as AGTransformer
        GraphDataset = AGDataset
        Transformer = AGTransformer

    # Forward sampling args so datasets can sample `n_samples_per_file` per JSON
    train_dataset = GraphDataset(
        args.data_dir,
        args.task,
        args.algorithm,
        "train",
        max_seq_length=args.max_seq_length,
        n_samples_per_file=getattr(args, 'n_samples_per_file', -1),
        sampling_seed=getattr(args, 'sampling_seed', None),
    )
    valid_dataset = GraphDataset(
        args.data_dir,
        args.task,
        args.algorithm,
        "valid",
        max_seq_length=args.max_seq_length,
        n_samples_per_file=getattr(args, 'n_samples_per_file', -1),
        sampling_seed=getattr(args, 'sampling_seed', None),
    )
    test_dataset = GraphDataset(
        args.data_dir,
        args.task,
        args.algorithm,
        "test",
        max_seq_length=args.max_seq_length,
        n_samples_per_file=getattr(args, 'n_samples_per_file', -1),
        sampling_seed=getattr(args, 'sampling_seed', None),
    )

    # Optionally limit dataset sizes via args (set in model config or CLI overrides)
    try:
        if getattr(args, 'max_samples_train', None) and args.max_samples_train > 0:
            n = min(len(train_dataset), int(args.max_samples_train))
            train_dataset = Subset(train_dataset, list(range(n)))
        if getattr(args, 'max_samples_valid', None) and args.max_samples_valid > 0:
            n = min(len(valid_dataset), int(args.max_samples_valid))
            valid_dataset = Subset(valid_dataset, list(range(n)))
        if getattr(args, 'max_samples_test', None) and args.max_samples_test > 0:
            n = min(len(test_dataset), int(args.max_samples_test))
            test_dataset = Subset(test_dataset, list(range(n)))
    except Exception:
        # Keep original datasets on any error
        pass

    # Share vocabulary if available. If datasets are wrapped in `Subset`,
    # extract the underlying base dataset to access `token2idx`/`vocab_size`.
    def _base(ds):
        try:
            from torch.utils.data import Subset as _Subset
            return ds.dataset if isinstance(ds, _Subset) else ds
        except Exception:
            return ds

    base_train = _base(train_dataset)
    base_valid = _base(valid_dataset)
    base_test = _base(test_dataset)

    try:
        if hasattr(base_train, 'token2idx'):
            base_valid.token2idx = base_train.token2idx
            base_valid.idx2token = base_train.idx2token
            base_valid.vocab_size = base_train.vocab_size
            # token-id mismatches between training and test time.
            base_test.token2idx = base_train.token2idx
            base_test.idx2token = base_train.idx2token
            base_test.vocab_size = base_train.vocab_size
    except Exception:
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Ensure we pass the underlying dataset's vocab_size (handle Subset wrappers).
    vocab_size = getattr(base_train, 'vocab_size', None)
    model = Transformer(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, max_seq_length=args.max_seq_length).to(device)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "model": model,
    }
