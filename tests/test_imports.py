def test_imports():
    import sm_system

    # smoke-test public surface
    assert hasattr(sm_system, "__version__")
    from sm_system import Simulation, SM_System  # noqa: F401
