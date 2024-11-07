from lamoto.training.auxiliary.callbacks import CallbackAtRatchetingInterval, EventType


def tst_ratcheting():
    # Expected: 10, 20, 30, 40, ..., 100, 200, 300, 400, ..., 1000, 2000, 3000, ...
    callback = CallbackAtRatchetingInterval(start=10, steps_between_ratchets=9, events=EventType.CHECKPOINT)
    for i in range(5001):
        if callback.should_event_happen(i):
            print(i)

    print()

    # Expected: 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 5000, ...
    callback = CallbackAtRatchetingInterval(start=20, steps_between_ratchets=4, events=EventType.CHECKPOINT)
    for i in range(5001):
        if callback.should_event_happen(i):
            print(i)


if __name__ == "__main__":
    tst_ratcheting()