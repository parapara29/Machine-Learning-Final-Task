import numpy as np

from robotnik_dagger_collect import sample_random_spawn, sample_random_target


def test_random_target_is_within_requested_ranges():
    rng = np.random.default_rng(7)
    target = sample_random_target(
        rng,
        x_range=(0.4, 1.2),
        y_range=(-1.0, 1.0),
        z_range=(0.2, 1.2),
    )

    assert 0.4 <= target[0] <= 1.2
    assert -1.0 <= target[1] <= 1.0
    assert 0.2 <= target[2] <= 1.2
    assert np.isfinite(target).all()


def test_random_spawn_is_within_requested_ranges():
    rng = np.random.default_rng(11)
    spawn = sample_random_spawn(
        rng,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        yaw_range=(-np.pi, np.pi),
    )

    assert -2.0 <= spawn[0] <= 2.0
    assert -2.0 <= spawn[1] <= 2.0
    assert -np.pi <= spawn[2] <= np.pi
    assert np.isfinite(spawn).all()


if __name__ == '__main__':   # also runnable without pytest
    test_random_target_is_within_requested_ranges()
    test_random_spawn_is_within_requested_ranges()
    print('sampler tests passed')
