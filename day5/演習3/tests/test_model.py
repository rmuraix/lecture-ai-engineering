import os
import pickle
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_preprocessing_output_shape(sample_data, preprocessor):
    """前処理後のデータ形状の確認"""
    X = sample_data.drop("Survived", axis=1)
    preprocessor.fit(X)
    transformed = preprocessor.transform(X)
    assert transformed.shape[0] == X.shape[0], "前処理後の行数が一致しません"


def test_model_with_missing_values(train_model, sample_data):
    """欠損値を含むデータに対するモデル推論テスト"""
    model, _, _ = train_model
    test_data = sample_data.drop("Survived", axis=1).iloc[:5].copy()
    test_data.iloc[0, 0] = np.nan  # 最初の行の最初の列に欠損値を導入

    try:
        _ = model.predict(test_data)
    except Exception as e:
        pytest.fail(f"欠損値を含むデータで推論中にエラーが発生: {e}")


def test_model_loading():
    """保存されたモデルが正しく読み込めるか"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        assert hasattr(
            model, "predict"
        ), "読み込んだモデルがpredictメソッドを持っていません"
    except Exception as e:
        pytest.fail(f"モデルの読み込みに失敗しました: {e}")


def test_model_with_invalid_input(train_model):
    """無効な入力に対するモデルの挙動を確認"""
    model, _, _ = train_model
    invalid_input = pd.DataFrame(
        {
            "Pclass": ["a"],
            "Sex": ["male"],
            "Age": ["?"],
            "SibSp": [0],
            "Parch": [0],
            "Fare": [7.25],
            "Embarked": ["S"],
        }
    )

    with pytest.raises(Exception):
        model.predict(invalid_input)


def test_model_with_empty_input(train_model):
    """空の入力に対するエラーハンドリング"""
    model, _, _ = train_model
    empty_input = pd.DataFrame(
        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    )

    with pytest.raises(ValueError):
        model.predict(empty_input)


def test_model_prediction_labels(train_model):
    """予測ラベルが想定範囲内であることの検証"""
    model, X_test, _ = train_model
    predictions = model.predict(X_test)
    assert set(predictions).issubset(
        {0, 1}
    ), f"予測値が0/1以外を含んでいます: {set(predictions)}"
