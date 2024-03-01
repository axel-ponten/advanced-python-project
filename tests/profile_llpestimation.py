import timeit
import cProfile
import pstats

# cProfile for the calculations
print("####### cProfile for probability calculation #######")
with cProfile.Profile() as profile:
    for i in range(1000):
        my_LLPEstimator.calc_llp_probability(length_list, energy_list)
profile_result = pstats.Stats(profile)
profile_result.sort_stats(pstats.SortKey.TIME)
profile_result.print_stats()
