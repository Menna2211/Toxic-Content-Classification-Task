from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def build_bilstm_model(max_features, max_length, num_classes):
    """Build BiLSTM model architecture"""
    model = Sequential([
        Embedding(max_features, 128, input_length=max_length),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("\nEnhanced BiLSTM Model Architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, y_encoded, epochs=100, batch_size=16, patience=10):
    """Train the model with early stopping and class weights"""
    # Compute class weights
    classes_for_cw = np.unique(y_encoded)
    cw = compute_class_weight(class_weight='balanced', classes=classes_for_cw, y=y_encoded)
    class_weight = {i: w for i, w in enumerate(cw)}
    print("Class weights:", class_weight)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print("\nTraining BiLSTM model with early stopping...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1,
        shuffle=True
    )
    
    print(f"\nTraining completed!")
    print(f"Total epochs trained: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    
    return history