from torch.utils.data import DataLoader


def build_mpnn(args, device):
    # Import inside function to avoid heavy imports at module-import time
    from mpnn import GraphTaskDataset, GIN, GraphMPNNTrainer, collate_fn

    train_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "train")
    valid_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "valid")
    test_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GIN(in_features=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, out_features=1, dropout=0.5)
    task_type = "classification" if args.task in ["cycle_check"] else "regression"
    trainer = GraphMPNNTrainer(model, learning_rate=args.learning_rate, device=device, task_type=task_type)

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

    train_dataset = GraphDataset(args.data_dir, args.task, args.algorithm, "train", max_seq_length=args.max_seq_length)
    valid_dataset = GraphDataset(args.data_dir, args.task, args.algorithm, "valid", max_seq_length=args.max_seq_length)
    test_dataset = GraphDataset(args.data_dir, args.task, args.algorithm, "test", max_seq_length=args.max_seq_length)

    # Share vocabulary if available
    try:
        valid_dataset.token2idx = train_dataset.token2idx
        valid_dataset.idx2token = train_dataset.idx2token
        valid_dataset.vocab_size = train_dataset.vocab_size
        # token-id mismatches between training and test time.
        test_dataset.token2idx = train_dataset.token2idx
        test_dataset.idx2token = train_dataset.idx2token
        test_dataset.vocab_size = train_dataset.vocab_size
    except Exception:
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Transformer(vocab_size=getattr(train_dataset, 'vocab_size', None), d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, max_seq_length=args.max_seq_length).to(device)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "model": model,
    }
