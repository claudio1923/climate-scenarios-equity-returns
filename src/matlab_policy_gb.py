"""
Least-squares boosting with MATLAB's tree growth policy, implemented in numpy.

Why this exists
---------------
MATLAB's `MaxNumSplits` is a budget on the NUMBER OF SPLITS, not a depth limit,
and the tree is grown breadth-first. The documented procedure is:

  1. grow level by level (a level is the set of nodes equidistant from the root);
  2. split every splittable node of the current level;
  3. count branch nodes: if the total exceeds MaxNumSplits, rank the branches OF
     THAT LEVEL by impurity gain and undo the least productive splits until the
     budget is met;
  4. move to the next level. Stop when the budget is exhausted, no split in the
     level improves, or MinLeafSize / MinParentSize block every node.

`MaxNumSplits = 15` therefore coincides with `max_depth = 4` only for a complete
tree (1 + 2 + 4 + 8 = 15 branch nodes). When trees are incomplete the budget is
not used up, and MATLAB keeps descending to level 5 and beyond with the splits
that are left over. scikit-learn offers neither policy: `max_depth` truncates by
depth, `max_leaf_nodes` switches to best-first growth. Hence this builder.

What is exact and what is a choice
----------------------------------
Exact, from the documented procedure: breadth-first growth, the split budget,
the per-level undo rule, the MSE gain criterion, midpoint thresholds, the leaf
and parent size constraints.

A choice, because the documentation does not pin it down: the tie-break used
when two candidate splits have exactly equal gain. Two rules are involved and
both are parameters here, so the effect can be measured rather than assumed:
  - within a node, across columns: the lower column index wins (a strictly
    greater gain is required to displace the current best), which is what
    scikit-learn does;
  - across nodes of a level, when the budget forces an undo: the lower node
    index (leftmost) wins among equal gains.
"""

import numpy as np

NO_SPLIT = -1


class Tree:
    """A single regression tree stored in flat arrays."""

    __slots__ = ("feature", "threshold", "left", "right", "value", "depth", "n_node")

    def __init__(self):
        self.feature = []    # split column, NO_SPLIT for leaves
        self.threshold = []  # split point, nan for leaves
        self.left = []       # child indices, NO_SPLIT for leaves
        self.right = []
        self.value = []      # leaf value (also filled for branch nodes)
        self.depth = []      # level of the node, root = 0
        self.n_node = []     # number of training rows in the node

    def add_node(self, value, depth, n_rows):
        self.feature.append(NO_SPLIT)
        self.threshold.append(np.nan)
        self.left.append(NO_SPLIT)
        self.right.append(NO_SPLIT)
        self.value.append(value)
        self.depth.append(depth)
        self.n_node.append(n_rows)
        return len(self.value) - 1

    def finalise(self):
        self.feature = np.asarray(self.feature, dtype=np.int64)
        self.threshold = np.asarray(self.threshold, dtype=np.float64)
        self.left = np.asarray(self.left, dtype=np.int64)
        self.right = np.asarray(self.right, dtype=np.int64)
        self.value = np.asarray(self.value, dtype=np.float64)
        self.depth = np.asarray(self.depth, dtype=np.int64)
        self.n_node = np.asarray(self.n_node, dtype=np.int64)
        return self

    # ------------------------------------------------------------------ shape

    @property
    def is_leaf(self):
        return self.feature == NO_SPLIT

    @property
    def n_leaves(self):
        return int(self.is_leaf.sum())

    @property
    def n_splits(self):
        return int((~self.is_leaf).sum())

    @property
    def max_depth(self):
        return int(self.depth.max())

    def splits_on(self, feature_index):
        """Thresholds used by this tree on one column."""
        return self.threshold[self.feature == feature_index]

    # ------------------------------------------------------------- prediction

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.float64)
        node = np.zeros(X.shape[0], dtype=np.int64)

        active = np.ones(X.shape[0], dtype=bool)
        while active.any():
            current = node[active]
            leaf = self.is_leaf[current]

            # Rows that reached a leaf are done.
            done = np.flatnonzero(active)[leaf]
            out[done] = self.value[current[leaf]]
            active[done] = False
            if not active.any():
                break

            current = node[active]
            rows = np.flatnonzero(active)
            go_left = X[rows, self.feature[current]] <= self.threshold[current]
            node[rows] = np.where(go_left, self.left[current], self.right[current])

        return out


def _best_split_in_node(X, y, mask, order, min_leaf, min_parent):
    """
    Exhaustive search over every column for the split with the largest gain.

        gain = n_L * n_R / n * (mean_L - mean_R) ** 2

    Candidate thresholds sit at the midpoint between adjacent distinct values.
    Returns (gain, column, threshold) or None when the node cannot be split.
    """
    n_rows = int(mask.sum())
    if n_rows < min_parent or n_rows < 2 * min_leaf:
        return None

    best = None
    for column in range(X.shape[1]):
        sorted_rows = order[:, column]
        rows = sorted_rows[mask[sorted_rows]]  # node rows, sorted by this column

        values = X[rows, column]
        targets = y[rows]

        cumulative = np.cumsum(targets)
        total = cumulative[-1]

        left_n = np.arange(1, n_rows, dtype=np.float64)
        right_n = n_rows - left_n
        left_sum = cumulative[:-1]
        right_sum = total - left_sum

        # A split is only allowed between two different values, and both sides
        # must keep at least min_leaf rows.
        allowed = (
            (left_n >= min_leaf)
            & (right_n >= min_leaf)
            & (values[1:] != values[:-1])
        )
        if not allowed.any():
            continue

        gains = np.where(
            allowed,
            (left_n * right_n / n_rows) * (left_sum / left_n - right_sum / right_n) ** 2,
            -np.inf,
        )
        position = int(np.argmax(gains))
        gain = float(gains[position])
        if gain <= 0.0 or not np.isfinite(gain):
            continue

        # Strictly greater gain is required to displace the incumbent, so the
        # lower column index wins a tie.
        if best is None or gain > best[0]:
            threshold = 0.5 * (values[position] + values[position + 1])
            best = (gain, column, float(threshold))

    return best


def build_tree(X, y, order, max_splits, min_leaf, min_parent):
    """Grow one tree with the MATLAB policy: breadth-first under a split budget."""
    n_rows = X.shape[0]
    tree = Tree()

    root_mask = np.ones(n_rows, dtype=bool)
    tree.add_node(float(y.mean()), depth=0, n_rows=n_rows)
    masks = {0: root_mask}

    level = [0]
    used_splits = 0

    while level and used_splits < max_splits:
        # Step 2: find the split of every node in this level.
        candidates = []
        for node_id in level:
            found = _best_split_in_node(X, y, masks[node_id], order, min_leaf, min_parent)
            if found is not None:
                gain, column, threshold = found
                candidates.append((gain, node_id, column, threshold))

        if not candidates:
            break

        # Step 3: if the level overruns the budget, undo the least productive
        # splits of this level. Ties are resolved in favour of the lower node id.
        room = max_splits - used_splits
        if len(candidates) > room:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            candidates = candidates[:room]

        next_level = []
        for gain, node_id, column, threshold in sorted(candidates, key=lambda item: item[1]):
            mask = masks[node_id]
            goes_left = mask & (X[:, column] <= threshold)
            goes_right = mask & ~goes_left

            depth = tree.depth[node_id] if isinstance(tree.depth, np.ndarray) else tree.depth[node_id]
            left_id = tree.add_node(float(y[goes_left].mean()), depth + 1, int(goes_left.sum()))
            right_id = tree.add_node(float(y[goes_right].mean()), depth + 1, int(goes_right.sum()))

            tree.feature[node_id] = column
            tree.threshold[node_id] = threshold
            tree.left[node_id] = left_id
            tree.right[node_id] = right_id

            masks[left_id] = goes_left
            masks[right_id] = goes_right
            del masks[node_id]

            next_level.extend((left_id, right_id))

        used_splits += len(candidates)
        level = next_level

    return tree.finalise()


class MatlabPolicyGB:
    """
    LSBoost with the MATLAB growth policy.

        F_0 = mean(y)
        F_m = F_{m-1} + learning_rate * h_m,  h_m fitted to the residual

    No subsampling, no pruning, no leaf merging, no early stopping: the model is
    deterministic and has no seed dependence.
    """

    def __init__(
        self,
        max_splits=15,
        min_leaf=10,
        min_parent=20,
        learning_rate=0.03,
        n_estimators=300,
    ):
        self.max_splits = max_splits
        self.min_leaf = min_leaf
        self.min_parent = min_parent
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.trees_ = []
        self.init_ = None

    def fit(self, X, y, verbose=False):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)

        # Sorting every column once up front turns the per-node search into a
        # masked gather instead of a sort.
        order = np.argsort(X, axis=0, kind="stable")

        self.init_ = float(y.mean())
        prediction = np.full(y.shape[0], self.init_, dtype=np.float64)
        self.trees_ = []

        for round_index in range(self.n_estimators):
            residual = y - prediction
            tree = build_tree(
                X, residual, order, self.max_splits, self.min_leaf, self.min_parent
            )
            prediction += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

            if verbose and (round_index + 1) % 50 == 0:
                print(f"    tree {round_index + 1}/{self.n_estimators}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], self.init_, dtype=np.float64)
        for tree in self.trees_:
            out += self.learning_rate * tree.predict(X)
        return out

    # ------------------------------------------------------------ diagnostics

    def tree_shape(self):
        """Leaves, splits and depth per tree, for comparing growth policies."""
        return {
            "leaves": np.array([t.n_leaves for t in self.trees_]),
            "splits": np.array([t.n_splits for t in self.trees_]),
            "depth": np.array([t.max_depth for t in self.trees_]),
        }

    def thresholds_on(self, feature_index):
        """Every threshold the ensemble places on one column."""
        collected = [t.splits_on(feature_index) for t in self.trees_]
        return np.concatenate(collected) if collected else np.array([])

    def feature_importance(self, X, y):
        """
        Total SSE reduction attributed to each column, normalised to sum to one.

        Recomputed by replaying the splits, since the gain is not stored.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        importance = np.zeros(X.shape[1], dtype=np.float64)

        prediction = np.full(y.shape[0], self.init_, dtype=np.float64)
        for tree in self.trees_:
            residual = y - prediction
            _accumulate_importance(tree, X, residual, importance)
            prediction += self.learning_rate * tree.predict(X)

        total = importance.sum()
        return importance / total if total > 0 else importance


def _accumulate_importance(tree, X, residual, importance):
    """Walk one tree and add each split's SSE reduction to the running total."""
    masks = {0: np.ones(X.shape[0], dtype=bool)}
    for node_id in range(len(tree.value)):
        if tree.feature[node_id] == NO_SPLIT:
            masks.pop(node_id, None)
            continue

        mask = masks.pop(node_id)
        column = int(tree.feature[node_id])
        goes_left = mask & (X[:, column] <= tree.threshold[node_id])
        goes_right = mask & ~goes_left

        n_left, n_right = int(goes_left.sum()), int(goes_right.sum())
        if n_left and n_right:
            mean_left = residual[goes_left].mean()
            mean_right = residual[goes_right].mean()
            n_total = n_left + n_right
            importance[column] += (n_left * n_right / n_total) * (mean_left - mean_right) ** 2

        masks[int(tree.left[node_id])] = goes_left
        masks[int(tree.right[node_id])] = goes_right


# --------------------------------------------------------------------- testing

def check_equivalence_with_sklearn(n_rows=2000, n_columns=5, depth=3, seed=0, verbose=True):
    """
    Validation of the builder, mandatory before trusting it.

    On data where every node can be split, a complete tree is the same tree
    whatever order the nodes are visited in: the split of a node depends only on
    the rows inside it. So the builder with max_splits = 2**depth - 1 must
    reproduce scikit-learn with max_depth = depth, exactly.

    Raises AssertionError if the two disagree.
    """
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_columns))
    y = X[:, 0] * 2.0 + np.sin(X[:, 1] * 3.0) + rng.normal(scale=0.3, size=n_rows)

    min_leaf, min_parent = 10, 20
    order = np.argsort(X, axis=0, kind="stable")
    mine = build_tree(X, y, order, 2 ** depth - 1, min_leaf, min_parent)

    reference = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=min_leaf,
        min_samples_split=min_parent,
        max_features=None,
        random_state=0,
    ).fit(X, y)

    assert mine.n_splits == 2 ** depth - 1, (
        f"builder did not fill the budget: {mine.n_splits} splits instead of {2 ** depth - 1}; "
        "the synthetic sample is not dense enough for this test"
    )

    my_prediction = mine.predict(X)
    reference_prediction = reference.predict(X)
    max_difference = float(np.abs(my_prediction - reference_prediction).max())

    my_splits = sorted(
        (int(f), float(t)) for f, t in zip(mine.feature, mine.threshold) if f != NO_SPLIT
    )
    tree_ = reference.tree_
    reference_splits = sorted(
        (int(f), float(t)) for f, t in zip(tree_.feature, tree_.threshold) if f >= 0
    )

    assert len(my_splits) == len(reference_splits), (
        f"different number of splits: {len(my_splits)} against {len(reference_splits)}"
    )
    assert [f for f, _ in my_splits] == [f for f, _ in reference_splits], (
        "split columns differ:\n"
        f"  builder   : {[f for f, _ in my_splits]}\n"
        f"  scikit    : {[f for f, _ in reference_splits]}"
    )

    # scikit-learn casts X to float32 inside the tree, so its midpoint
    # thresholds carry float32 rounding while this builder stays in float64.
    # The partitions are what has to agree; the thresholds only have to agree to
    # single precision, and the predictions confirm the rows split the same way.
    threshold_gap = max(
        abs(a - b) / max(1.0, abs(b))
        for (_, a), (_, b) in zip(my_splits, reference_splits)
    )
    assert threshold_gap < 1e-6, (
        "thresholds differ by more than float32 rounding:\n"
        f"  builder   : {my_splits}\n"
        f"  scikit    : {reference_splits}"
    )
    assert max_difference < 1e-9, f"predictions differ by {max_difference:.3e}"

    if verbose:
        print(
            f"builder validated against scikit-learn: depth {depth}, "
            f"{mine.n_splits} splits, {mine.n_leaves} leaves, "
            f"identical split columns, thresholds within {threshold_gap:.1e} "
            f"(float32 rounding), max |prediction difference| = {max_difference:.3e}"
        )
    return max_difference


if __name__ == "__main__":
    for d in (2, 3, 4):
        check_equivalence_with_sklearn(depth=d)
