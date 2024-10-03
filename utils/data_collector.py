import os
from collections.abc import Iterable

import h5py
import numpy as np
import torch


class DataCollector:

    def __init__(self, env_name: str, directory_path: str, filename: str):
        # save input arguments
        self._env_name = env_name
        self._filename = filename
        self._directory = os.path.abspath(directory_path)

        # placeholder for current hdf5 file object
        self._h5_file_stream = None
        self._h5_data_group = None
        self._h5_episode_group = None

        self._flush_freq = 1
        self._num_demos = 1

        # create directory it doesn't exist
        if not os.path.isdir(self._directory):
            os.makedirs(self._directory)

        # flags for setting up
        self._is_first_interaction = True
        self._is_stop = False
        # create buffers to store data
        self._dataset = dict()

    def is_stopped(self) -> bool:
        """Whether data collection is stopped or not.

        Returns:
            True if data collection has stopped.
        """
        return self._is_stop

    def reset(self):
        """Reset the internals of data logger."""
        # setup the file to store data in
        if self._is_first_interaction:
            self._demo_count = 0
            self._create_new_file(self._filename)
            self._is_first_interaction = False
        # clear out existing buffers
        self._dataset = dict()

    def add(self, key: str, value: np.ndarray | torch.Tensor):

        if self._is_first_interaction:
            print("Please call reset before adding new data. Calling reset...")
            self.reset()
        if self._is_stop:
            print(f"Collecting data is stopped")
            return
        # check datatype
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        else:
            value = np.asarray(value)
        # check if there are sub-keys
        sub_keys = key.split("/")
        num_sub_keys = len(sub_keys)
        if len(sub_keys) > 2:
            raise ValueError(
                f"Input key '{key}' has elements {num_sub_keys} which is more than two."
            )
        # add key to dictionary if it doesn't exist
        for i in range(value.shape[0]):
            # demo index
            if f"env_{i}" not in self._dataset:
                self._dataset[f"env_{i}"] = dict()
            # key index
            if num_sub_keys == 2:
                # create keys
                if sub_keys[0] not in self._dataset[f"env_{i}"]:
                    self._dataset[f"env_{i}"][sub_keys[0]] = dict()
                if sub_keys[1] not in self._dataset[f"env_{i}"][sub_keys[0]]:
                    self._dataset[f"env_{i}"][sub_keys[0]][sub_keys[1]] = list()
                # add data to key
                self._dataset[f"env_{i}"][sub_keys[0]][sub_keys[1]].append(value[i])
            else:
                # create keys
                if sub_keys[0] not in self._dataset[f"env_{i}"]:
                    self._dataset[f"env_{i}"][sub_keys[0]] = list()
                # add data to key
                self._dataset[f"env_{i}"][sub_keys[0]].append(value[i])

    def flush(self, env_ids: Iterable[int] = (0,)):
        """Flush the episode data based on environment indices.

        Args:
            env_ids: Environment indices to write data for. Defaults to (0).
        """
        # check that data is being recorded
        if self._h5_file_stream is None or self._h5_data_group is None:
            print(
                "No file stream has been opened. Please call reset before flushing data."
            )
            return

        # iterate over each environment and add their data
        for index in env_ids:
            # data corresponding to demo
            env_dataset = self._dataset[f"env_{index}"]

            # create episode group based on demo count
            h5_episode_group = self._h5_data_group.create_group(
                f"demo_{self._demo_count}"
            )
            # store number of steps taken
            h5_episode_group.attrs["num_samples"] = len(env_dataset["actions"])
            # store other data from dictionary
            for key, value in env_dataset.items():
                if isinstance(value, dict):
                    # create group
                    key_group = h5_episode_group.create_group(key)
                    # add sub-keys values
                    for sub_key, sub_value in value.items():
                        key_group.create_dataset(sub_key, data=np.array(sub_value))
                else:
                    h5_episode_group.create_dataset(key, data=np.array(value))
            # increment total step counts
            self._h5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

            # increment total demo counts
            self._demo_count += 1
            # reset buffer for environment
            self._dataset[f"env_{index}"] = dict()

            # dump at desired frequency
            if self._demo_count % self._flush_freq == 0:
                self._h5_file_stream.flush()
                print(
                    f">>> Flushing data to disk. Collected demos: {self._demo_count} / {self._num_demos}"
                )

            # if demos collected then stop
            if self._demo_count >= self._num_demos:
                print(
                    f">>> Desired number of demonstrations collected: {self._demo_count} >= {self._num_demos}."
                )
                self.close()
                # break out of loop
                break

    def close(self):
        """Stop recording and save the file at its current state."""
        if not self._is_stop:
            # close the file safely
            if self._h5_file_stream is not None:
                self._h5_file_stream.close()
            # mark that data collection is stopped
            self._is_stop = True

    def _create_new_file(self, fname: str):
        """Create a new HDF5 file for writing episode info into.

        Reference:
            https://robomimic.github.io/docs/datasets/overview.html

        Args:
            fname: The base name of the file.
        """
        if not fname.endswith(".hdf5"):
            fname += ".hdf5"
        # define path to file
        hdf5_path = os.path.join(self._directory, fname)
        # construct the stream object
        self._h5_file_stream = h5py.File(hdf5_path, "w")
        # create group to store data
        self._h5_data_group = self._h5_file_stream.create_group("data")
        # stores total number of samples accumulated across demonstrations
        self._h5_data_group.attrs["total"] = 0
        # store the environment meta-info
        # -- we use gym environment type
        # Ref: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_base.py#L15
        # env_type = 2
        # -- check if env config provided
        # if self._env_config is None:
        # self._env_config = dict()
        # -- add info
        # self._h5_data_group.attrs["env_args"] = json.dumps({
        #     "env_name": self._env_name,
        #     "type": env_type,
        #     "env_kwargs": self._env_config,
        # })
