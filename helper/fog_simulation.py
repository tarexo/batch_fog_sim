import os
import copy
import pickle
import numpy as np
from pathlib import Path
from numpy.random import Generator
from typing import Dict, List, Tuple
from classes.parameter_set import ParameterSet


class FogSimulation:
    def __init__(
        self, INTEGRAL_PATH: Path, AVAILABLE_TAU_Hs: list[int], RNG: Generator
    ):
        self.INTEGRAL_PATH = INTEGRAL_PATH
        self.AVAILABLE_TAU_Hs = AVAILABLE_TAU_Hs
        self.RNG = RNG

    def get_available_alphas(self) -> List[float]:
        alphas = []

        for file in os.listdir(self.INTEGRAL_PATH):
            if file.endswith(".pickle"):
                alpha = file.split("_")[-1].replace(".pickle", "")

                alphas.append(float(alpha))

        return sorted(alphas)

    def get_integral_dict(self, p: ParameterSet) -> Dict:
        alphas = self.get_available_alphas()

        alpha = min(alphas, key=lambda x: abs(x - p.alpha))
        tau_h = min(self.AVAILABLE_TAU_Hs, key=lambda x: abs(x - int(p.tau_h * 1e9)))

        filename = (
            self.INTEGRAL_PATH
            / f"integral_0m_to_200m_stepsize_0.1m_tau_h_{tau_h}ns_alpha_{alpha}.pickle"
        )

        with open(filename, "rb") as handle:
            integral_dict = pickle.load(handle)

        return integral_dict

    def P_R_fog_hard(self, p: ParameterSet, pc: np.ndarray) -> np.ndarray:
        r_0 = np.linalg.norm(pc[:, 0:3], axis=1)

        pc[:, 3] = np.round(np.exp(-2 * p.alpha * r_0) * pc[:, 3])

        return pc

    def P_R_fog_soft(
        self,
        p: ParameterSet,
        pc: np.ndarray,
        original_intesity: np.ndarray,
        noise: int,
        gain: bool = False,
        noise_variant: str = "v1",
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        augmented_pc = np.zeros(pc.shape)
        fog_mask = np.zeros(len(pc), dtype=bool)

        r_zeros = np.linalg.norm(pc[:, 0:3], axis=1)

        min_fog_response = np.inf
        max_fog_response = 0
        num_fog_responses = 0

        integral_dict = self.get_integral_dict(p)

        r_noise = self.RNG.integers(low=1, high=20, size=1)[0]
        r_noise = 10

        for i, r_0 in enumerate(r_zeros):
            # load integral values from precomputed dict
            key = float(str(round(r_0, 1)))
            # limit key to a maximum of 200 m
            fog_distance, fog_response = integral_dict[min(key, 200)]

            fog_response = (
                fog_response * original_intesity[i] * (r_0**2) * p.beta / p.beta_0
            )

            # limit to 255
            fog_response = min(fog_response, 255)

            if fog_response > pc[i, 3]:
                fog_mask[i] = 1

                num_fog_responses += 1

                scaling_factor = fog_distance / r_0

                augmented_pc[i, 0] = pc[i, 0] * scaling_factor
                augmented_pc[i, 1] = pc[i, 1] * scaling_factor
                augmented_pc[i, 2] = pc[i, 2] * scaling_factor
                augmented_pc[i, 3] = fog_response

                # keep 5th feature if it exists
                if pc.shape[1] > 4:
                    augmented_pc[i, 4] = pc[i, 4]

                if noise > 0:
                    if noise_variant == "v1":
                        # add uniform noise based on initial distance
                        distance_noise = self.RNG.uniform(
                            low=r_0 - noise, high=r_0 + noise, size=1
                        )[0]
                        noise_factor = r_0 / distance_noise

                    elif noise_variant == "v2":
                        # add noise in the power domain
                        power = self.RNG.uniform(low=-1, high=1, size=1)[0]
                        noise_factor = (
                            max(1.0, noise / 5) ** power
                        )  # noise=10 => noise_factor ranges from 1/2 to 2

                    elif noise_variant == "v3":
                        # add noise in the power domain
                        power = self.RNG.uniform(low=-0.5, high=1, size=1)[0]
                        noise_factor = (
                            max(1.0, noise * 4 / 10) ** power
                        )  # noise=10 => ranges from 1/2 to 4

                    elif noise_variant == "v4":
                        additive = r_noise * self.RNG.beta(a=2, b=20, size=1)[0]
                        new_dist = fog_distance + additive
                        noise_factor = new_dist / fog_distance

                    else:
                        raise NotImplementedError(
                            f"noise variant '{noise_variant}' is not implemented (yet)"
                        )

                    augmented_pc[i, 0] = augmented_pc[i, 0] * noise_factor
                    augmented_pc[i, 1] = augmented_pc[i, 1] * noise_factor
                    augmented_pc[i, 2] = augmented_pc[i, 2] * noise_factor

                if fog_response > max_fog_response:
                    max_fog_response = fog_response

                if fog_response < min_fog_response:
                    min_fog_response = fog_response

            else:
                augmented_pc[i] = pc[i]

        if gain:
            max_intensity = np.ceil(max(augmented_pc[:, 3]))
            gain_factor = 255 / max_intensity
            augmented_pc[:, 3] *= gain_factor

        simulated_fog_pc = None

        if num_fog_responses > 0:
            fog_points = augmented_pc[fog_mask]
            simulated_fog_pc = fog_points

        info_dict = {
            "min_fog_response": min_fog_response,
            "max_fog_response": max_fog_response,
            "num_fog_responses": num_fog_responses,
        }

        return augmented_pc, simulated_fog_pc, info_dict

    def simulate_fog(
        self,
        p: ParameterSet,
        pc: np.ndarray,
        noise: int,
        gain: bool = False,
        noise_variant: str = "v1",
        hard: bool = True,
        soft: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        augmented_pc = copy.deepcopy(pc)
        original_intensity = copy.deepcopy(pc[:, 3])

        info_dict = None
        simulated_fog_pc = None

        if hard:
            augmented_pc = self.P_R_fog_hard(p, augmented_pc)
        if soft:
            augmented_pc, simulated_fog_pc, info_dict = self.P_R_fog_soft(
                p, augmented_pc, original_intensity, noise, gain, noise_variant
            )

        return augmented_pc, simulated_fog_pc, info_dict
