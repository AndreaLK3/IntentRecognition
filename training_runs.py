import Model.Training as T

# (batch_size, learning_rate)
hyperparams_lts = [(2,5e-5), (4,1e-4), (4,5e-4), (4,1e-3), (8,2e-4), (8,5e-4), (8,1e-3)]


if __name__ == "__main__":

    for (bsz, lr) in hyperparams_lts:
        model = T.run_train(learning_rate=lr, batch_size=bsz)