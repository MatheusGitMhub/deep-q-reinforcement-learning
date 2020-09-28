from tqdm import tqdm
from reinforcement_learning_module import (AI_Trader,
                                           stocks_price_format,
                                           dataset_loader,
                                           state_creator)

# Cargar data de la divisa de entrenamiento
stock_name = "AAPL"
data = dataset_loader(stock_name)

# Definir hiper parámetros del IATrader
window_size = 10
episodes = 1000
batch_size = 32
data_samples = len(data) - 1

# Definir el modelo
trader = AI_Trader(window_size)
print(trader.model.summary())


# Entrenamiento por reinforcement learning
for episode in range(1, episodes + 1):

    print(f"Episodio: {episode}/{episodes}")
    state = state_creator(data, 0, window_size + 1)
    total_profit = 0
    trader.inventory = []

    for t in tqdm(range(data_samples)):
        action = trader.trade(state)
        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0

        # Comprar
        if action == 1:
            trader.inventory.append(data[t])
            print("AI Trader compró: ", stocks_price_format(data[t]))

        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)
            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print("AI Trader vendió: ",
                  stocks_price_format(data[t]),
                  " Beneficio: " + stocks_price_format(data[t] - buy_price))

        if t == data_samples - 1:
            done = True
        else:
            done = False

        trader.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("########################")
            print(f"BENEFICIO TOTAL: {total_profit}")
            print("########################")
        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    if episode % 10 == 0:
        trader.model.save(f"ai_trader_{episode}.h5")

print(f"Terminaron los {episodes} de entrenamiento")