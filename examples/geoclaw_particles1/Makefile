# Makefile for Clawpack code in this directory.
# This version only sets the local files and frequently changed
# options, and then includes the standard makefile pointed to by CLAWMAKE.
CLAWMAKE = $(CLAW)/clawutil/src/Makefile.common

# See the above file for details and a list of make options, or type
#   make .help
# at the unix prompt.


# Adjust these variables if desired:
# ----------------------------------

CLAW_PKG = geoclaw                  # Clawpack package to use
EXE = xgeoclaw                      # Executable to create
SETRUN_FILE = setrun.py             # File containing function to make data
OUTDIR = _output                    # Directory for output
SETPLOT_FILE = setplot.py           # File containing function to set plots
PLOTDIR = _plots                    # Directory for plots


# Environment variable FC should be set to fortran compiler, e.g. gfortran

# Compiler flags can be specified here or set as an environment variable
FFLAGS ?= -O2 -fopenmp
OMP_NUM_THREADS ?= 4

# ---------------------------------
# package sources for this program:
# ---------------------------------

AMRLIB = $(CLAW)/amrclaw/src/2d
GEOLIB = $(CLAW)/geoclaw/src/2d/shallow
include $(GEOLIB)/Makefile.geoclaw

NEWAMR = ../../new_fortran_amrclaw
NEWGEO = ../../new_fortran_geoclaw

# ---------------------------------------
# package sources specifically to exclude
# (i.e. if a custom replacement source 
#  under a different name is provided)
# ---------------------------------------

EXCLUDE_MODULES = \
  $(GEOLIB)/topo_module.f90 \
  $(GEOLIB)/qinit_module.f90 \
  $(GEOLIB)/fgmax_module.f90 \
  $(AMRLIB)/regions_module.f90 \

EXCLUDE_SOURCES = \

# ----------------------------------------
# List of custom sources for this program:
# ----------------------------------------


MODULES = \
  $(NEWGEO)/topo_module.f90 \
  $(NEWGEO)/qinit_module.f90 \
  $(NEWGEO)/fgmax_module.f90 \
  $(NEWAMR)/regions_module.f90 \

SOURCES = \
  $(NEWAMR)/flagregions2.f90 \
  $(NEWGEO)/filpatch.f90 \
  $(NEWGEO)/filval.f90 \
  $(NEWGEO)/set_eta_init_dtopo.f90 \
  $(NEWGEO)/fgmax_read.f90 \
  $(NEWGEO)/fgmax_finalize.f90 \
  $(NEWGEO)/fgmax_interpolate0.f90 \
  $(NEWGEO)/fgmax_values.f90 \
  $(NEWGEO)/fgmax_frompatch.f90 \
  $(NEWGEO)/tick.f \
  $(NEWGEO)/qinit.f90 \
  $(NEWGEO)/flag2refine2.f90 \
  $(CLAW)/riemann/src/rpn2_geoclaw.f \
  $(CLAW)/riemann/src/rpt2_geoclaw.f \
  $(CLAW)/riemann/src/geoclaw_riemann_utils.f \


#-------------------------------------------------------------------
# Include Makefile containing standard definitions and make options:
include $(CLAWMAKE)

# Construct the topography data
.PHONY: topo all
topo:
	python maketopo.py

all: 
	$(MAKE) topo
	$(MAKE) .plots
	$(MAKE) .htmls

