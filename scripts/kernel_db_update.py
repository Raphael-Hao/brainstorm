from brt.jit.codegen.storage import KernelStorager
import json
from typing import Dict, List
from collections import OrderedDict

kernel_storager = KernelStorager()
cursor = kernel_storager.cursor

SELECT_ALL_CMD = r"""
SELECT Key, Identifier, OpType 
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
for key, id, optype in qrs:
    id: Dict = json.loads(id)

    # # Delete parameters.module_name
    # popret = id['parameters'].pop('module_name', None)
    # if popret is not None:
    #     print(id['parameters'])
    #     id = json.dumps(id)
    #     cursor.execute(UPDATE_CMD, (id, key))
    #     cursor.connection.commit()
        
    # Update output size of ConvTranspose2d
    # {
    #     "output_infos": {
    #         "output_0": [34, 3, 32, 32]
    #     },
    #     ...
    # }
    if optype == 'ConvTranspose2dBias':
        outinfo: List = id['output_infos']['output_0']
        if outinfo is None:
            id['output_infos']['output_0'] = [id['input_infos']['input_0'][0], 3, 128, 128]
        else:
            outinfo[:2].extend([128, 128])
            id['output_infos']['output_0'] = outinfo
        print(f"{id['output_infos']['output_0']}")

    
    # Sort id
    id['input_infos'] = OrderedDict(sorted(id['input_infos'].items()))
    id['output_infos'] = OrderedDict(sorted(id['output_infos'].items()))
    id['parameters'] = OrderedDict(sorted(id['parameters'].items()))
    id = OrderedDict(sorted(id.items()))


    id = json.dumps(id)
    # print(id)
    cursor.execute(UPDATE_CMD, (id, key))
    cursor.connection.commit()
        

