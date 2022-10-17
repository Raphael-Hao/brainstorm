# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import sqlite3

from brt.runtime import BRT_KERNEL_DB_FNAME

__all__ = ["kernel_storager"]


class KernelStorager:
    QUERY_KERNEL_CMD = r"""
SELECT Key, Identifier, OpType, Attributes, Source, DeviceType, Function, Tags, Miscs
FROM KernelCache
WHERE (Identifier = ?) AND (DeviceType = ?) AND (ObjectiveFunc = ?) AND (Rank = ?);"""
    ADD_KERNEL_CMD = r"""
INSERT INTO KernelCache
    (Key, Identifier, OpType, Attributes, Source, DeviceType, Function, Tags, Miscs, ObjectiveFunc, Rank) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    DEL_KERNEL_CMD = r"""
DELETE FROM KernelCache
WHERE (Identifier = ?) AND (ObjectiveFunc = ?) AND (Rank = ?);"""
    INIT_DB_CMD = r"""
CREATE TABLE IF NOT EXISTS KernelCache(
   Key           TEXT NOT NULL,
   Identifier    TEXT NOT NULL,
   OpType        TEXT NOT NULL,
   Attributes    TEXT DEFAULT "",
   Source        TEXT DEFAULT "External",
   DeviceType    TEXT NOT NULL,
   Function      TEXT NOT NULL,
   Tags          TEXT DEFAULT "",
   Miscs         TEXT DEFAULT "",
   ObjectiveFunc TEXT DEFAULT "fastest",
   Rank          INT  DEFAULT 1,
   PRIMARY KEY(Identifier, ObjectiveFunc, Rank)
   );
"""

    def __init__(self) -> None:
        """KernelStorager is a class to store kernel information in a database.
        The kernel storager is consistent with nnfusion for now.
        Key: generated according to input shape, kernel name
        Identifier:
        OpType:
        Attributes:
        Source: brt
        DeviceType: GPU in brt
        Function: kernel code in json string format including fucntion_siganature, function_body, and function_dep

        Tags:
        Miscs:
        """
        self.db_path = BRT_KERNEL_DB_FNAME
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.init_kernel_cache_db()
        self.model_kernels = []

    def add_kernel(
        self,
        kernel_dict,
        overwrite=False,
    ):
        if overwrite:
            self.cursor.execute(
                self.DEL_KERNEL_CMD,
                (
                    kernel_dict["Identifier"],
                    kernel_dict["ObjectiveFunc"],
                    kernel_dict["Rank"],
                ),
            )

        self.cursor.execute(
            self.ADD_KERNEL_CMD,
            (
                kernel_dict["Key"],
                kernel_dict["Identifier"],
                kernel_dict["OpType"],
                kernel_dict["Attributes"],
                kernel_dict["Source"],
                kernel_dict["DeviceType"],
                kernel_dict["Function"],
                kernel_dict["Tags"],
                kernel_dict["Miscs"],
                kernel_dict["ObjectiveFunc"],
                kernel_dict["Rank"],
            ),
        )
        self.flush()

    def init_kernel_cache_db(self):
        self.cursor.execute(self.INIT_DB_CMD)

    def query_kernel(
        self,
        kernel_identifier,
        device_type,
        objective_func: str = "fastest",
        rank: int = 1,
    ):
        self.cursor.execute(
            self.QUERY_KERNEL_CMD,
            (kernel_identifier, device_type, objective_func, rank),
        )
        return self.cursor.fetchone()

    def flush(self):
        self.connection.commit()


kernel_storager = KernelStorager()
