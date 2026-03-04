from collections import OrderedDict
from .client import Client
from attackers import get_attacker_handler
from datapreprocessor.data_utils import subset_by_idx
from attackers import data_poisoning_attacks, hybrid_attacks
from evaluation import evaluate_multiclass_metrics, aggregate_client_metrics


def init_clients(args, client_train_indices, client_val_indices, train_dataset, test_dataset):
    clients = []
    for worker_id in range(args.num_clients):
        # for attacker, if the attack type is not model poisoning attack, use the default client class. For data poisoning attacks, it's already handled in the client class.
        # for benign clients, use the default client class
        if args.attack == "NoAttack":
            """
            For NoAttack scenario, use client class, and ignore args.num_adv
            """
            client_obj = Client
        else:
            if args.num_adv == 0:
                raise AssertionError(
                    "Attack {args.attack} specified, but attackers set to 0.")
            client_obj = Client if worker_id >= args.num_adv else get_attacker_handler(
                args.attack)
        local_dataset = subset_by_idx(
            args, train_dataset, client_train_indices[worker_id], train=True)
        local_val_dataset = subset_by_idx(
            args, train_dataset, client_val_indices[worker_id], train=False)
        tmp_client = client_obj(args, worker_id,
                                local_dataset, test_dataset)
        tmp_client.val_dataset = local_val_dataset
        clients.append(tmp_client)
    return clients


def set_fl_algorithm(args, the_server, clients):
    """set the federated learning algorithm for the server and clients. If the algorithm type is not specified in arguments, use the default algorithm type of the server.

    Args:
        the_server (Server): server object
        clients (Client): a list of client objects
        algorithm (str): federated learning algorithm types

    Raises:
        ValueError: No specified or default algorithm type can be used
    """
    if args.algorithm:
        alg_type = args.algorithm
    elif hasattr(the_server, 'algorithm'):
        args.algorithm = the_server.algorithm
    else:
        raise ValueError(
            "No specified algorithm or default algorithm type of the server. Please specify an algorithm type, with `--algorithm`")

    the_server.set_algorithm(alg_type)
    for client in clients:
        client.set_algorithm(alg_type)


def evaluate(the_server, test_dataset, args, global_epoch):
    """
    Backdoor attacks evaluation requires inference-time attacks. However, since the server is unaware of the backdoor attacks, the client's `client_test` is used in the coordinator for ASR evaluation.
    """
    test_keys = ["Test Acc", "Test loss", "ASR", "ASR loss"]
    results = OrderedDict()

    # normal evaluation
    imbalanced_flag = True if 'imbalanced' in args.distribution else False
    if imbalanced_flag:
        test_keys.insert(1, 'Tail Acc')

    test_loader = the_server.get_dataloader(test_dataset, train_flag=False)
    clean_test = the_server.test(
        the_server.global_model, test_loader, imbalanced=imbalanced_flag)

    for idx in range(len(clean_test)):
        results[test_keys[idx]] = clean_test[idx]

    if args.attack in data_poisoning_attacks + hybrid_attacks:
        # index [0, f] is poisoning attacker
        results['ASR'], results['ASR loss'] = the_server.clients[0].client_test(
            the_server.global_model, test_dataset, poison_epochs=True)
    return results


def evaluate_clients_validation(clients):
    val_accs, val_losses = [], []
    for client in clients:
        val_acc, val_loss = client.validate()
        val_accs.append(val_acc)
        val_losses.append(val_loss)
    return val_accs, val_losses


def final_evaluate_from_best_checkpoints(clients, test_dataset, args):
    """
    Final evaluation based on each client's best local validation checkpoint.
    """
    clean_per_client = []
    asr_per_client = []

    attack_eval_client = None
    if args.attack in data_poisoning_attacks + hybrid_attacks:
        attack_eval_client = next(
            (c for c in clients if getattr(c, "category", None) == "attacker"),
            None
        )

    for client in clients:
        client.load_best_checkpoint()
        clean_loader = client.get_dataloader(test_dataset, train_flag=False)
        clean_metrics = evaluate_multiclass_metrics(
            client.model, clean_loader, args.device, args.num_classes
        )
        clean_metrics["best_epoch"] = float(client.best_epoch)
        clean_metrics["best_val_acc"] = float(client.best_val_acc)
        clean_per_client.append(clean_metrics)

        if attack_eval_client is not None:
            asr, asr_loss = attack_eval_client.client_test(
                model=client.model,
                test_dataset=test_dataset,
                poison_epochs=True
            )
            asr_per_client.append({"asr": float(asr), "asr_loss": float(asr_loss)})
        else:
            asr_per_client.append({"asr": float("nan"), "asr_loss": float("nan")})

    clean_summary = aggregate_client_metrics(clean_per_client)
    asr_summary = aggregate_client_metrics(asr_per_client)

    return {
        "clean_per_client": clean_per_client,
        "asr_per_client": asr_per_client,
        "clean_summary": clean_summary,
        "asr_summary": asr_summary,
    }
