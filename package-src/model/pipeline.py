from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from model.config.core import config


categorical_features = ["Gender"]

numeric_features = [
    feature
    for feature in config.model_config.features
    if feature not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", MinMaxScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

abandono_pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=config.model_config.n_estimators,
                max_depth=config.model_config.max_depth,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
