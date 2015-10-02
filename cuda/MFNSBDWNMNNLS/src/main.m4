#Copyright 2015 Merlin Kramer
#Licensed under the GNU Affero General Public License v3.0
#include <stdio.h>
#include <stdlib.h>
#include <cufftXt.h>
#include <assert.h>
#include <sm_30_intrinsics.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <driver_functions.h>
define(`concatVarSize', eval(`1 ** 8'))
