import gc
import logging
import os
import time
from fl import coordinator
from global_args import benchmark_preprocess, read_args, override_args, single_preprocess
from global_utils import avg_value, print_filtered_args, setup_logger, setup_seed
from datapreprocessor.data_utils import (
    load_data,
    split_dataset,
    reassign_adversary_indices,
    split_client_train_val_indices,
)
from fl.server import Server
from plot_utils import plot_accuracy, plot_convergence, plot_label_distribution, visualize_dba_triggers
from attackers import data_poisoning_attacks, hybrid_attacks


def fl_run(args):
    """
    function to run federated learning logics
    """
    # setup logger
    args.logger = setup_logger(
        __name__, f'{args.output}', level=logging.INFO)
    print_filtered_args(args, args.logger)
    start_time = time.time()
    args.logger.info(
        f"Started on {time.asctime(time.localtime(start_time))}")
    run_tag = os.path.splitext(os.path.basename(args.output))[0]
    # fix randomness
    setup_seed(args.seed)

    # 1. load dataset and split dataset indices for clients with i.i.d or non-i.i.d
    train_dataset, test_dataset = load_data(args)
    client_indices, test_dataset = split_dataset(
        args, train_dataset, test_dataset)
    args.logger.info("Data partitioned")

    # reassign adversary slots to clients best suited for the attack
    client_indices = reassign_adversary_indices(
        args, train_dataset, client_indices)
    client_train_indices, client_val_indices = split_client_train_val_indices(
        client_indices, train_ratio=7, val_ratio=1
    )
    total_train = sum(len(x) for x in client_train_indices)
    total_val = sum(len(x) for x in client_val_indices)
    args.logger.info(
        f"Client train/val split ready (train={total_train}, val={total_val}, ratio=7/1)")

    # visualize non-IID label distribution
    if args.distribution == "non-iid":
        output_dir = os.path.dirname(args.output)
        alpha_val = getattr(args, 'dirichlet_alpha', None)
        plot_label_distribution(
            train_dataset, client_indices, args.num_clients,
            args.dataset, args.distribution,
            output_dir=output_dir, num_adv=args.num_adv,
            dirichlet_alpha=alpha_val, filename_prefix=run_tag)
        args.logger.info(f"Non-IID distribution chart saved to {output_dir}")

    # 2. initialize clients and server with seperate training data indices
    clients = coordinator.init_clients(
        args, client_train_indices, client_val_indices, train_dataset, test_dataset)
    the_server = Server(args, clients, test_dataset, train_dataset)

    # visualize DBA triggers
    if args.attack == "DBA" and args.attack in data_poisoning_attacks + hybrid_attacks:
        visualize_dba_triggers(clients[0], args, filename_prefix=run_tag)
        args.logger.info(f"DBA trigger visualization saved")

    # 3. initialize the federated learning algorithm for clients and server
    coordinator.set_fl_algorithm(args, the_server, clients)
    args.logger.info("Clients and server are initialized")
    args.logger.info("Starting Training...")
    for global_epoch in range(args.epochs):
        epoch_msg = f"Epoch {global_epoch:<3}\t"
        # print(f"Global epoch {global_epoch} begin")
        # server dispatches numpy version global weights 1d vector to clients
        global_weights_vec = the_server.global_weights_vec

        # clients' local training
        avg_train_acc, avg_train_loss = [], []
        avg_val_acc, avg_val_loss = [], []
        for client in clients:
            client.load_global_model(global_weights_vec)
            train_acc, train_loss = client.local_training()
            val_acc, val_loss = client.validate()
            client.maybe_update_best_checkpoint(val_acc, global_epoch)
            client.fetch_updates()
            avg_train_acc.append(train_acc)
            avg_train_loss.append(train_loss)
            avg_val_acc.append(val_acc)
            avg_val_loss.append(val_loss)

        avg_train_loss = avg_value(avg_train_loss)
        avg_train_acc = avg_value(avg_train_acc)
        valid_val_acc = [x for x in avg_val_acc if x == x]
        valid_val_loss = [x for x in avg_val_loss if x == x]
        mean_val_acc = avg_value(valid_val_acc) if valid_val_acc else float("nan")
        mean_val_loss = avg_value(valid_val_loss) if valid_val_loss else float("nan")
        epoch_msg += (
            f"\tTrain Acc: {avg_train_acc:.4f}\tTrain loss: {avg_train_loss:.4f}\t"
            f"Val Acc: {mean_val_acc:.4f}\tVal loss: {mean_val_loss:.4f}\t"
        )

        # perform post-training attacks, for omniscient model poisoning attack, pass all clients
        omniscient_attack(clients)

        # server collects weights from clients
        the_server.collect_updates(global_epoch)
        the_server.aggregation()
        the_server.update_global()

        # print the training and validation results of the current global_epoch
        args.logger.info(epoch_msg)
        # clear memory
        gc.collect()

    if args.record_time:
        report_time(clients, the_server)

    final_stats = coordinator.final_evaluate_from_best_checkpoints(
        clients, test_dataset, args
    )
    for idx, metrics in enumerate(final_stats["clean_per_client"]):
        args.logger.info(
            f"Final Client {idx:02d} | best_epoch={int(metrics['best_epoch'])} "
            f"best_val_acc={metrics['best_val_acc']:.4f} "
            f"test_acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f} test_loss={metrics['loss']:.4f} "
            f"asr={final_stats['asr_per_client'][idx]['asr']:.4f} "
            f"asr_loss={final_stats['asr_per_client'][idx]['asr_loss']:.4f}"
        )

    clean_summary = final_stats["clean_summary"]
    asr_summary = final_stats["asr_summary"]
    args.logger.info(
        "Final Summary | "
        f"test_acc_mean={clean_summary.get('acc_mean', float('nan')):.4f} "
        f"test_acc_std={clean_summary.get('acc_std', float('nan')):.4f} "
        f"precision_mean={clean_summary.get('precision_mean', float('nan')):.4f} "
        f"recall_mean={clean_summary.get('recall_mean', float('nan')):.4f} "
        f"f1_mean={clean_summary.get('f1_mean', float('nan')):.4f} "
        f"auc_mean={clean_summary.get('auc_mean', float('nan')):.4f} "
        f"test_loss_mean={clean_summary.get('loss_mean', float('nan')):.4f} "
        f"asr_mean={asr_summary.get('asr_mean', float('nan')):.4f} "
        f"asr_std={asr_summary.get('asr_std', float('nan')):.4f} "
        f"asr_loss_mean={asr_summary.get('asr_loss_mean', float('nan')):.4f}"
    )

    plot_accuracy(args.output)
    plot_convergence(args.output)

    end_time = time.time()
    time_difference = end_time - start_time
    minutes, seconds = int(
        time_difference // 60), int(time_difference % 60)
    args.logger.info(
        f"Training finished on {time.asctime(time.localtime(end_time))} using {minutes} minutes and {seconds} seconds in total.")


def report_time(clients, the_server):
    [c.time_recorder.report(f"Client {idx}") for idx, c in enumerate(clients)]
    the_server.time_recorder.report("Server")


def omniscient_attack(clients):
    """
    Perform an omniscient attack, which involves eavesdropping or collusion
    between malicious clients to craft adversarial updates.
    """
    # Filter out all omniscient attackers from the client list
    omniscient_attackers = [
        client for client in clients
        if client.category == "attacker" and "omniscient" in client.attributes
    ]

    # If no omniscient attackers exist, exit early
    if not omniscient_attackers:
        return
    # Generate malicious updates using the first attacker's logic
    malicious_updates = omniscient_attackers[0].omniscient(clients)
    if malicious_updates is None:
        raise ValueError("No updates generated by the omniscient attacker")

    # Check if the malicious update is a single vector or a batch of updates
    is_single_update = len(
        malicious_updates.shape) == 1 or malicious_updates.shape[0] == 1

    if is_single_update:
        # If a single update is provided, all attackers perform their own attack
        omniscient_attackers[0].update = malicious_updates
        for client in omniscient_attackers[1:]:
            client.update = client.omniscient(clients)
    else:
        # If multiple updates are provided, assign each update to an attacker
        # An attack method aiming to provide the same updates for all attackers can return repeated updates.
        for client, update in zip(omniscient_attackers, malicious_updates):
            client.update = update


def main(args, cli_args):
    """
    preprocess the arguments, logics, and run the federated learning process
    """
    # if Benchmarks is True, run all combinations of attacks and defenses
    if cli_args.benchmark:
        benchmark_preprocess(args)
        fl_run(args)
    else:
        override_args(args, cli_args)
        single_preprocess(args)
        fl_run(args)


if __name__ == "__main__":
    args, cli_args = read_args()
    main(args, cli_args)
