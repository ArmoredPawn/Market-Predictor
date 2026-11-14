"""The model zoo: 40+ classification algorithms that compete in the tournament.

Every model answers the same question ("will the price be higher in h days?")
from the same clues, and is tested with the same walk-forward procedure. They
span the main families of machine learning, from one-line rules of thumb to
neural networks, so the tournament shows whether *any* approach finds a
reliable pattern.

Settings deliberately favour stability (strong regularization, shallow trees,
large leaves), because stock data is mostly noise and flexible models
memorize noise. Every model is single-threaded so the tournament itself can
run models in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

FAMILIES = (
    "Simple rules",
    "Linear models",
    "Probability models",
    "Nearest neighbors",
    "Kernel methods",
    "Trees & forests",
    "Boosting",
    "Neural networks",
    "Teams",
)


class RuleClassifier(ClassifierMixin, BaseEstimator):
    """A rule of thumb based on one clue, e.g. 'up if last month was up'.

    It learns nothing: the probability is a squashed version of the clue,
    flipped when `sign` is −1 (for contrarian rules).
    """

    def __init__(self, feature: str = "ret_21", sign: float = 1.0, scale: float = 0.05):
        self.feature = feature
        self.sign = sign
        self.scale = scale

    def fit(self, X, y):
        self.classes_ = np.array([0.0, 1.0])
        cols = list(getattr(X, "columns", []))
        self.col_ = cols.index(self.feature) if self.feature in cols else 0
        return self

    def predict_proba(self, X):
        x = np.asarray(X, dtype=float)[:, self.col_]
        p = 1.0 / (1.0 + np.exp(-self.sign * x / self.scale))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(float)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: str
    blurb: str  # one plain-English sentence
    factory: Callable[[], object]
    baseline: bool = False  # simple rules aren't "learning" models
    members: tuple[str, ...] = ()  # for teams: the models whose probabilities are averaged
    cost: int = 1  # rough relative training time, used to schedule the slowest models first

    def make(self):
        return self.factory()


def _scaled(*steps):
    return make_pipeline(StandardScaler(), *steps)


def _lr(c=0.05, **kw):
    return LogisticRegression(C=c, max_iter=2000, **kw)


def _team3():
    return VotingClassifier(
        [("lr", _scaled(_lr())), ("rf", ZOO["extra_trees"].make()), ("gb", ZOO["hgb"].make())], voting="soft"
    )


def _team5():
    return VotingClassifier(
        [
            ("lr", _scaled(_lr())),
            ("rf", ZOO["forest_shallow"].make()),
            ("gb", ZOO["hgb"].make()),
            ("knn", ZOO["knn_100"].make()),
            ("nb", ZOO["gnb"].make()),
        ],
        voting="soft",
    )


_SPECS = [
    # ---- Simple rules (baselines)
    ModelSpec("always_up", "Always guess the usual direction", "Simple rules",
              "Predicts whatever happened most often in the training data (usually 'up').",
              lambda: DummyClassifier(strategy="most_frequent"), baseline=True),
    ModelSpec("coin_flip", "Random guessing", "Simple rules",
              "Flips a weighted coin. Any real model must beat this.",
              lambda: DummyClassifier(strategy="stratified", random_state=0), baseline=True),
    ModelSpec("momentum_rule", "Momentum rule", "Simple rules",
              "Up if the price rose over the last month. 'The trend is your friend.'",
              lambda: RuleClassifier("ret_21", 1.0, 0.05), baseline=True),
    ModelSpec("trend_rule", "Long-term trend rule", "Simple rules",
              "Up if the price is above its 200-day average, a classic trader's signal.",
              lambda: RuleClassifier("dist_sma_200", 1.0, 0.05), baseline=True),
    ModelSpec("reversal_rule", "Bounce-back rule", "Simple rules",
              "Up if the price fell last week. Bets that short drops reverse.",
              lambda: RuleClassifier("ret_5", -1.0, 0.03), baseline=True),
    # ---- Linear models
    ModelSpec("logistic", "Logistic regression", "Linear models",
              "Weighs each clue and adds them up into a probability. The classic starting point.",
              lambda: _scaled(_lr())),
    ModelSpec("logistic_l1", "Lasso logistic regression", "Linear models",
              "Logistic regression that switches off unhelpful clues entirely.",
              lambda: _scaled(_lr(0.02, l1_ratio=1.0, solver="saga"))),
    ModelSpec("logistic_en", "Elastic-net logistic regression", "Linear models",
              "A blend of lasso and ridge penalties: drops some clues, shrinks the rest.",
              lambda: _scaled(_lr(0.05, l1_ratio=0.5, solver="saga"))),
    ModelSpec("logistic_weak", "Lightly-regularized logistic regression", "Linear models",
              "Logistic regression allowed to fit the data much more closely (risk of overfitting).",
              lambda: _scaled(_lr(10.0))),
    ModelSpec("ridge", "Ridge classifier", "Linear models",
              "Least-squares line fitting with a penalty on large weights.",
              lambda: _scaled(RidgeClassifier(alpha=10.0))),
    ModelSpec("lda", "Linear discriminant analysis", "Linear models",
              "Finds the straight-line boundary that best separates up days from down days.",
              lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ModelSpec("sgd_log", "Stochastic gradient descent (logistic)", "Linear models",
              "Logistic regression trained one example at a time, like many big-data systems.",
              lambda: _scaled(SGDClassifier(loss="log_loss", alpha=0.01, random_state=0))),
    ModelSpec("sgd_huber", "Robust linear classifier", "Linear models",
              "A linear model using a loss that ignores extreme outliers.",
              lambda: _scaled(SGDClassifier(loss="modified_huber", alpha=0.01, random_state=0))),
    ModelSpec("perceptron", "Perceptron", "Linear models",
              "The original 1958 neural network: a single artificial neuron.",
              lambda: _scaled(Perceptron(penalty="l2", alpha=0.001, random_state=0))),
    ModelSpec("linear_svm", "Linear support vector machine", "Linear models",
              "Draws the straight-line boundary with the widest safety margin.",
              lambda: _scaled(LinearSVC(C=0.01))),
    ModelSpec("pca_logistic", "PCA + logistic regression", "Linear models",
              "Compresses the clues into 5 summary signals first, then uses logistic regression.",
              lambda: _scaled(PCA(n_components=5, random_state=0), _lr())),
    ModelSpec("select_logistic", "Top-5 clues + logistic regression", "Linear models",
              "Keeps only the 5 clues most related to the answer, then uses logistic regression.",
              lambda: _scaled(SelectKBest(f_classif, k=5), _lr())),
    ModelSpec("poly_logistic", "Polynomial logistic regression", "Linear models",
              "Adds every pair of clues multiplied together, so it can capture interactions.",
              lambda: _scaled(PolynomialFeatures(2, include_bias=False), StandardScaler(), _lr(0.005))),
    ModelSpec("spline_logistic", "Spline logistic regression", "Linear models",
              "Lets each clue have a curved, not just straight-line, effect.",
              lambda: _scaled(SplineTransformer(n_knots=4, degree=2), _lr(0.02))),
    # ---- Probability models
    ModelSpec("gnb", "Gaussian naive Bayes", "Probability models",
              "Applies Bayes' rule, treating every clue as an independent bell curve.",
              lambda: GaussianNB()),
    ModelSpec("bnb", "Bernoulli naive Bayes", "Probability models",
              "Bayes' rule on yes/no versions of each clue (above or below average).",
              lambda: _scaled(BernoulliNB(binarize=0.0))),
    ModelSpec("qda", "Quadratic discriminant analysis", "Probability models",
              "Models up days and down days as two different clouds and asks which one today fits.",
              lambda: QuadraticDiscriminantAnalysis(reg_param=0.5)),
    # ---- Nearest neighbors
    ModelSpec("knn_20", "20 nearest neighbors", "Nearest neighbors",
              "Finds the 20 past days most similar to today and copies what happened next.",
              lambda: _scaled(KNeighborsClassifier(20))),
    ModelSpec("knn_100", "100 nearest neighbors", "Nearest neighbors",
              "Same idea with 100 similar days: steadier, less detailed.",
              lambda: _scaled(KNeighborsClassifier(100))),
    ModelSpec("knn_300_w", "300 nearest neighbors (weighted)", "Nearest neighbors",
              "Looks at 300 similar days, giving the most similar ones more say.",
              lambda: _scaled(KNeighborsClassifier(300, weights="distance"))),
    ModelSpec("centroid", "Nearest centroid", "Nearest neighbors",
              "Compares today with the 'average up day' and the 'average down day'.",
              lambda: _scaled(NearestCentroid())),
    # ---- Kernel methods
    ModelSpec("svm_rbf", "Support vector machine (RBF kernel)", "Kernel methods",
              "Draws a curvy boundary between up and down days using the 'kernel trick'.",
              lambda: _scaled(SVC(C=0.5, kernel="rbf", gamma="scale", cache_size=500)), cost=6),
    ModelSpec("svm_poly", "Support vector machine (polynomial kernel)", "Kernel methods",
              "A support vector machine whose boundary is a smooth polynomial curve.",
              lambda: _scaled(SVC(C=0.5, kernel="poly", degree=2, cache_size=500)), cost=5),
    ModelSpec("nystroem", "Kernel approximation + logistic regression", "Kernel methods",
              "Approximates a kernel machine with 150 random features, which is much faster.",
              lambda: _scaled(Nystroem(n_components=150, random_state=0), _lr(0.05))),
    ModelSpec("rff", "Random Fourier features + logistic regression", "Kernel methods",
              "Turns clues into 150 random wave patterns, then fits logistic regression.",
              lambda: _scaled(RBFSampler(n_components=150, gamma=0.02, random_state=0), _lr(0.05))),
    # ---- Trees & forests
    ModelSpec("tree_3", "Decision tree (3 levels)", "Trees & forests",
              "A flowchart of up to 3 yes/no questions about the clues.",
              lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=0)),
    ModelSpec("tree_6", "Decision tree (6 levels)", "Trees & forests",
              "A deeper flowchart with up to 6 questions: more detail, more risk of memorizing.",
              lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, random_state=0)),
    ModelSpec("forest_shallow", "Random forest", "Trees & forests",
              "100 small decision trees, each trained on random data, voting together.",
              lambda: RandomForestClassifier(60, max_depth=4, min_samples_leaf=50, max_samples=0.6,
                                             n_jobs=1, random_state=0), cost=4),
    ModelSpec("forest_deep", "Random forest (deep trees)", "Trees & forests",
              "A forest of bigger trees that can learn finer patterns.",
              lambda: RandomForestClassifier(60, max_depth=10, min_samples_leaf=20, max_samples=0.6,
                                             n_jobs=1, random_state=0), cost=5),
    ModelSpec("extra_trees", "Extra-trees forest", "Trees & forests",
              "Like a random forest, but the trees split at random points too: very fast and steady.",
              lambda: ExtraTreesClassifier(100, max_depth=6, min_samples_leaf=40, n_jobs=1, random_state=0), cost=4),
    ModelSpec("bagged_trees", "Bagged decision trees", "Trees & forests",
              "30 medium-sized trees, each trained on a random resample of the history.",
              lambda: BaggingClassifier(DecisionTreeClassifier(max_depth=5, min_samples_leaf=30), 30,
                                        max_samples=0.6, n_jobs=1, random_state=0), cost=5),
    # ---- Boosting
    ModelSpec("adaboost", "AdaBoost", "Boosting",
              "Builds tiny trees one after another, each focusing on the previous ones' mistakes.",
              lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50,
                                         learning_rate=0.5, random_state=0), cost=7),
    ModelSpec("gbm", "Gradient boosting", "Boosting",
              "Adds small trees one at a time, each correcting the remaining errors.",
              lambda: GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=0.08,
                                                 subsample=0.7, random_state=0), cost=7),
    ModelSpec("hgb", "Histogram gradient boosting", "Boosting",
              "A fast modern boosting method, similar to LightGBM and XGBoost.",
              lambda: HistGradientBoostingClassifier(learning_rate=0.08, max_iter=100, max_depth=3,
                                                     min_samples_leaf=60, l2_regularization=1.0, random_state=0), cost=3),
    ModelSpec("hgb_cautious", "Cautious gradient boosting", "Boosting",
              "Boosting with very small steps and big leaves, so it learns only strong patterns.",
              lambda: HistGradientBoostingClassifier(learning_rate=0.03, max_iter=80, max_depth=2,
                                                     min_samples_leaf=150, l2_regularization=5.0, random_state=0), cost=2),
    # ---- Neural networks
    ModelSpec("mlp_small", "Small neural network", "Neural networks",
              "One layer of 16 artificial neurons, trained with backpropagation.",
              lambda: _scaled(MLPClassifier((16,), alpha=1.0, max_iter=200, early_stopping=True, random_state=0)), cost=3),
    ModelSpec("mlp_deep", "Deeper neural network", "Neural networks",
              "Two layers (32 then 16 neurons) that can combine clues in complex ways.",
              lambda: _scaled(MLPClassifier((32, 16), alpha=1.0, max_iter=200, early_stopping=True, random_state=0)), cost=4),
    ModelSpec("mlp_wide", "Wide neural network (heavily regularized)", "Neural networks",
              "64 neurons held back by a strong penalty, so it can't just memorize.",
              lambda: _scaled(MLPClassifier((64,), alpha=10.0, max_iter=200, early_stopping=True, random_state=0)), cost=4),
    # ---- Teams
    ModelSpec("ensemble", "Team of 3", "Teams",
              "Averages logistic regression, an extra-trees forest and gradient boosting.",
              _team3, members=("logistic", "extra_trees", "hgb"), cost=8),
    ModelSpec("team5", "Team of 5", "Teams",
              "Averages five very different models: linear, forest, boosting, neighbors and Bayes.",
              _team5, members=("logistic", "forest_shallow", "hgb", "knn_100", "gnb"), cost=8),
]

ZOO: dict[str, ModelSpec] = {s.key: s for s in _SPECS}
LEARNING_MODELS = [k for k, s in ZOO.items() if not s.baseline]


def has_probabilities(model) -> bool:
    """False for models that output scores rather than calibrated probabilities."""
    try:
        return hasattr(model, "predict_proba") and callable(model.predict_proba)
    except AttributeError:
        return False
