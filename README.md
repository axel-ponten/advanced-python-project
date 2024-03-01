# advanced-python-project

## Topic

My project will be to develop and improve the code of my current research project, which is to estimate the rate of a rare hypothezised process at the IceCube neutrino observatory. The final product is a package with sphynx documentation that has a pytest script, profiling script and some simple application script.

## Background
This project is part of a beyond standard model analysis at the IceCube neutrino observatory, which is a cube kilometer large neutrino detector buried 2 km deep in the antarctic ice (https://icecube.wisc.edu/). The detector is built to measure rare cosmic neutrino interactions a few times a year, but also detects the abundant atmospheric muon background at a rate of 3 kHz. The analysis searches for signatures of a hypothesized class of exotic particles called long lived particles (LLPs) in the background atmospheric muon flux. The LLPs could potentially be produced in a bremstrahlung-like process where the muon scatters of an oxygen or hydrogen in the ice. The LLP would then decay after some time, hopefully inside the detector in order to be detectable.

IceCube measures particles through their Cherenkov radiation in the ice. LLPs are dark and do not emit any Cherenkov light. Their production and subsequent decay would therefore leave a gap in the muon track where there is no light emitted. As such, we need both the production and decay vertex of the LLP to be inside the detector volume in order for the event to be detectable.

## What does the package ***llpestimation*** do?

The package main class ***llpestimator*** computes the probability for a given atmospheric muon (represented by its energy along a list of length steps through the detector) to produce an LLP which has both production and decay vertex inside the detector. It does so for a list of LLP models that are defined by their name, mass, coupling, lifetime and production cross section (generated from interpolation tables). The production cross sections are pre-calculated since earlier and represented in tables that will be interpolated. The production also depend on the medium in which the muon travels, particularly the number density of oxygen and hydrogen in the south pole ice (for now we only consider oxygen).