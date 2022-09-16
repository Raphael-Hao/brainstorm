from python.brt.jit.codegen.storage import KernelStorager
import json
from typing import Dict
from collections import OrderedDict

kernel_storager = KernelStorager()
cursor = kernel_storager.cursor

SELECT_ALL_CMD = r"""
SELECT Key, Identifier
FROM   KernelCache"""
SELECT_KEY_CMD = r"""
SELECT Identifier
FROM   KernelCache
WHERE Key = ?"""
UPDATE_CMD = r"""
UPDATE KernelCache
SET Identifier = ?
WHERE Key = ?
"""
qrs = cursor.execute(SELECT_ALL_CMD).fetchall()
for key, id in qrs:
    id: Dict = json.loads(id)
    id['input_infos'] = OrderedDict(sorted(id['input_infos'].items()))
    id['output_infos'] = OrderedDict(sorted(id['output_infos'].items()))
    id['parameters'] = OrderedDict(sorted(id['parameters'].items()))
    id = OrderedDict(sorted(id.items()))
    id = json.dumps(id)
    print(id)
    cursor.execute(UPDATE_CMD, (id, key))
    cursor.connection.commit()

    # popret = id['parameters'].pop('module_name', None)
    # if popret is not None:
    #     print(id['parameters'])
    #     id = json.dumps(id)
    #     cursor.execute(UPDATE_CMD, (id, key))
    #     cursor.connection.commit()
        