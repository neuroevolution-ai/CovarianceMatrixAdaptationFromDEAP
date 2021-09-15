## Quick start

You need our slightly modified Deap version for testing. This modified Deap version exposes some extra states of the CMA-ES algorithm, but the calculation itself does not differ from the original version. To install our modified Deap version do the following: 

1. Uninstall the deap package if already installed via `pip uninstall deap`
2. Run `pip install git+https://github.com/neuroevolution-ai/deap@test-cma-es-in-julia` 
3. Then the testing should work
