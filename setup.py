from setuptools import setup

if __name__ == "__main__":
    setup(name="lorenz_gan",
          version="0.1",
          description="Testbed for stochastic parameterizations of the Lorenz '96 model.",
          author="David John Gagne",
          author_email="dgagne@ucar.edu",
          license="MIT",
          url="https://github.com/djgagne/lorenz_gan",
          packages=["lorenz_gan"],
          install_requires=["numpy",
                            "scipy",
                            "matplotlib",
                            "xarray",
                            "netcdf4",
                            "tensorflow<=1.15.1",
                            "keras",
                            "numba",
                            "pandas",
                            "jupyter",
                            "scikit-learn",
                            "pyyaml"])
