from src.prediction.predictor import load_signals, load_model, predict_next
signals = load_signals()
model = load_model()
print(predict_next(signals, model))
