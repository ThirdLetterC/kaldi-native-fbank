/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 */

/*! \file pocketfft.h
 *  Public interface of the pocketfft library
 *
 *  Copyright (C) 2008-2018 Max-Planck-Society
 *  \author Martin Reinecke
 */

#pragma once

#include <stddef.h>

struct cfft_plan_i;
typedef struct cfft_plan_i *cfft_plan;
/* Returns an owning complex FFT plan. Caller must release with
 * destroy_cfft_plan. */
[[nodiscard]] cfft_plan make_cfft_plan(size_t length);
void destroy_cfft_plan(cfft_plan plan);
[[nodiscard]] int cfft_backward(cfft_plan plan, double c[], double fct);
[[nodiscard]] int cfft_forward(cfft_plan plan, double c[], double fct);
[[nodiscard]] size_t cfft_length(cfft_plan plan);

struct rfft_plan_i;
typedef struct rfft_plan_i *rfft_plan;
/* Returns an owning real FFT plan. Caller must release with destroy_rfft_plan.
 */
[[nodiscard]] rfft_plan make_rfft_plan(size_t length);
void destroy_rfft_plan(rfft_plan plan);
[[nodiscard]] int rfft_backward(rfft_plan plan, double c[], double fct);
[[nodiscard]] int rfft_forward(rfft_plan plan, double c[], double fct);
[[nodiscard]] size_t rfft_length(rfft_plan plan);
