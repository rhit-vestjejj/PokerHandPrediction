import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
import tensorflow as tf

cols = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "H"]

df = pd.read_csv("poker-hand-training-true.data", names = cols)

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

def plot_loss(history):
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Binary crossnetropy")
    plt.legend()
    plt.grid(True)

    plt.show()

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = "relu", input_shape = (10,)),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])

y = tf.keras.utils.to_categorical(y, num_classes=10)

# 50 ~ 0.68   500 ~ 0.95   5000 ~ 0.999
history = nn_model.fit(X, y, epochs = 50, batch_size = 64, validation_split=0.2)

# plot_loss(history)

df2 = pd.read_csv("poker-hand-testing.data", names = cols)

test, valid = np.split(df2.sample(frac=1), [int(0.5 * len(df))])

test = df2[df2.columns[:-1]].values
y_test = df2[df2.columns[-1]].values

pred = nn_model.predict(test)
pred = pred.astype(int).reshape(-1,)

while True:
    try:
        print("Format of the string is Suit Card so H 12 for heart queen, etc")
        card1 = input("What is card 1?")
        
        card1_list = card1.split(" ")
        if card1_list[0] == "H":
            S1 = 1
        elif card1_list[0] == "S":
            S1 = 2
        elif card1_list[0] == "D":
            S1 = 3
        elif card1_list[0] == "C":
            S1 = 4
        else:
            raise ValueError("Put a valid card")
        if int(card1_list[1]) <= 13 and int(card1_list[1]) >= 1:
            C1 = int(card1_list[1])
        else:
            raise ValueError("Put a valid card")
        
        card2 = input("What is card 2?")
        
        card2_list = card2.split(" ")
        if card2_list[0] == "H":
            S2 = 1
        elif card2_list[0] == "S":
            S2 = 2
        elif card2_list[0] == "D":
            S2 = 3
        elif card2_list[0] == "C":
            S2 = 4
        else:
            raise ValueError("Put a valid card")
        if int(card2_list[1]) <= 13 and int(card2_list[1]) >= 1:
            C2 = int(card2_list[1])
        else:
            raise ValueError("Put a valid card")
        
        card3 = input("What is card 3?")

        card3_list = card3.split(" ")
        if card3_list[0] == "H":
            S3 = 1
        elif card3_list[0] == "S":
            S3 = 2
        elif card3_list[0] == "D":
            S3 = 3
        elif card3_list[0] == "C":
            S3 = 4
        else:
            raise ValueError("Put a valid card")
        if int(card3_list[1]) <= 13 and int(card3_list[1]) >= 1:
            C3 = int(card3_list[1])
        else:
            raise ValueError("Put a valid card")
        
        card4 = input("What is card 4?")

        card4_list = card4.split(" ")
        if card4_list[0] == "H":
            S4 = 1
        elif card4_list[0] == "S":
            S4 = 2
        elif card4_list[0] == "D":
            S4 = 3
        elif card4_list[0] == "C":
            S4 = 4
        else:
            raise ValueError("Put a valid card")
        if int(card4_list[1]) <= 13 and int(card4_list[1]) >= 1:
            C4 = int(card4_list[1])
        else:
            raise ValueError("Put a valid card")
        
        card5 = input("What is card 5?")
        
        card5_list = card5.split(" ")
        if card5_list[0] == "H":
            S5 = 1
        elif card5_list[0] == "S":
            S5 = 2
        elif card5_list[0] == "D":
            S5 = 3
        elif card5_list[0] == "C":
            S5 = 4
        else:
            raise ValueError("Put a valid card")
        if int(card5_list[1]) <= 13 and int(card5_list[1]) >= 1:
            C5 = int(card5_list[1])
        else:
            raise ValueError("Put a valid card")



        test_input = [S1, C1, S2, C2, S3, C3, S4, C4, S5, C5]
        test_input = np.array(test_input).reshape(1, -1)

        prediction = nn_model.predict(test_input)

        data = [0.50117739, 0.42256903, 0.04753902, 0.02112845, 0.00392464, 0.0019654, 0.00144058, 0.0002401, 0.00001385, 0.00000154]
        possible_hands = ["Nothing", "One pair", "Two pair", "Three of a kind", "Straight", "Flush", "Full house", "Four of a kind", "Straight flush", "Royal flush"]

        total = 0

        for i in range(len(data)):
            prediction[0][i] = prediction[0][i] / data[i]
            total += prediction[0][i]
        
        dict_data = {
            
        }
        
        highest = 0
        index = 0
        
        for i in range(len(prediction[0])):
            if prediction[0][i] > highest:
                index = i
                highest = prediction[0][i]
            dict_data[possible_hands[i]] = (prediction[0][i] / total).astype(float)

        dict_data = {key: float(round(value, 7)) for key, value in dict_data.items()}
        dict_data = dict(sorted(dict_data.items(), key=lambda item: item[1], reverse=True))

        for key in dict_data:
            print(f"{key} : {dict_data[key]}")

    except Exception as e:
        print(e)