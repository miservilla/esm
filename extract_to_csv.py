#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Run on cli: python extract.py esm2_t33_650M_UR50D ../TC_FASTA.fasta outputs --repr_layers 33 --include per_tok --truncation_seq_length 5026

import argparse
import pathlib
import torch
import csv  # Import CSV module

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Open a new CSV file for writing embeddings
    csv_file_path = args.output_dir / "FASTA_embeddings_20240213.csv"
    with csv_file_path.open(mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write CSV header (if necessary)
        # csv_writer.writerow(['label', 'embedding_layer_1', 'embedding_layer_2', ...])

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                if torch.cuda.is_available() and not args.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                out = model(toks, repr_layers=args.repr_layers)

                for i, label in enumerate(labels):
                    # Extract desired embeddings here and format them for CSV writing
                    embedding_data = [label]  # Start with the label
                    for layer in args.repr_layers:
                        embeddings = out["representations"][layer][i].cpu().numpy()
                        # Handle embedding data as needed, for example, by flattening or summarizing
                        embedding_summary = embeddings.mean(axis=0)  # Example: mean over tokens
                        embedding_data.extend(embedding_summary.tolist())

                    # Write the processed embedding data for the current sequence to the CSV file
                    csv_writer.writerow(embedding_data)


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
