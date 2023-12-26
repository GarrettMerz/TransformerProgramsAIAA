import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):
    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/scratch/4loop_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)
    # embed ######################################################
    embed_df = pd.read_csv("output/scratch/4loop_embeddings.csv").set_index("word")
    embeddings = embed_df.loc[tokens]
    var0_embeddings = embeddings["var0_embeddings"].tolist()
    var0_embedding_scores = classifier_weights.loc[
        [("var0_embeddings", str(v)) for v in var0_embeddings]
    ]

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 0
        elif q_position in {2, 3, 4, 8, 9}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 6

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, var0_embeddings)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 0
        elif q_position in {2, 3}:
            return k_position == 6
        elif q_position in {8, 9, 4}:
            return k_position == 7
        elif q_position in {5, 6, 7}:
            return k_position == 3

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, var0_embeddings)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(var0_embedding, position):
        if var0_embedding in {0, 1, 2, 3, 4, 8}:
            return position == 5
        elif var0_embedding in {9, 5, 6, 7}:
            return position == 4

    num_attn_0_0_pattern = select(positions, var0_embeddings, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(var0_embedding, position):
        if var0_embedding in {0}:
            return position == 9
        elif var0_embedding in {1}:
            return position == 8
        elif var0_embedding in {2, 4, 7, 8, 9}:
            return position == 3
        elif var0_embedding in {3}:
            return position == 2
        elif var0_embedding in {5, 6}:
            return position == 5

    num_attn_0_1_pattern = select(positions, var0_embeddings, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 7),
            (0, 8),
            (0, 9),
            (2, 0),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 7),
            (3, 8),
            (3, 9),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 8),
            (4, 9),
            (7, 0),
            (7, 3),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 8),
            (8, 9),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 8),
            (9, 9),
        }:
            return 6
        elif key in {(3, 5), (5, 3)}:
            return 3
        return 2

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_var0_embedding):
        key = (num_attn_0_0_output, num_var0_embedding)
        return 0

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1) for k0, k1 in zip(num_attn_0_0_outputs, num_var0_embeddings)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                var0_embedding_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                mlp_0_0_output_scores,
                num_mlp_0_0_output_scores,
                num_var0_embedding_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "c", "e", "c", "c", "a", "f", "f", "f", "</s>"]))
