from logreg import *
import matplotlib.pyplot as plt

def question1():
  etas = {
  .001: [],
  .01: [],
  .1: [],
  1: [],
  10: [],
  100: [],
  1000: []}

  passes = 10

  train, test, vocab = read_dataset('../data/hockey_baseball/positive',
    '../data/hockey_baseball/negative', '../data/hockey_baseball/vocab')

  for eta in etas:
    # Initialize model
    lr = LogReg(len(vocab), 0.0, lambda x: eta)

    # Iterations
    iteration = 0
    for pp in xrange(passes):
        random.shuffle(train)
        for ex in train:
            lr.sg_update(ex, iteration)
            if iteration % 1000 == 1:
                # train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                etas[eta].append(ho_lp)
                # print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      # (iteration, train_lp, ho_lp, train_acc, ho_acc))
            iteration += 1

  for eta in sorted(etas.iterkeys()):
    y = etas[eta]
    x = range(0, len(y))
    plt.plot(x, y, label=str(eta))
  plt.legend(loc='lower right')
  plt.title('Convergence of log-likelihood function value given value of eta')
  plt.ylabel('Test set log-likelihood function value')
  plt.xlabel('Thousand iteration')
  plt.savefig('q1.png')


def main():
  question1()

if __name__ == '__main__':
  main()