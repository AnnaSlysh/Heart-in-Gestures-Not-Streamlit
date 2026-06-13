"""
Extract dense layer weights from TFLite model using flatbuffers.
Saves model_weights.json for use in browser JS inference.
"""
import json, numpy as np, sys, os, struct
import flatbuffers
from flatbuffers import number_types as N

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
OUT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'keypoint_classifier', 'model_weights.json')


def read_tflite_raw(path):
    """Low-level TFLite reader. Uses raw flatbuffers table navigation."""
    with open(path, 'rb') as f:
        data = bytearray(f.read())
    buf = flatbuffers.encode.Get(N.UOffsetTFlags.packer_type, data, 0)
    # Root object offset (the uint32 at byte 0 is an offset from byte 0)
    return data, buf


def _vtable_field(buf, table_pos, field_id):
    """Get the absolute position of a field value within a table, or None."""
    vtable_soffset = int.from_bytes(buf[table_pos:table_pos+4], 'little', signed=True)
    vtable_pos = table_pos - vtable_soffset
    vtable_size = int.from_bytes(buf[vtable_pos:vtable_pos+2], 'little')
    field_offset_in_vtable = 4 + field_id * 2
    if field_offset_in_vtable >= vtable_size:
        return None
    field_rel = int.from_bytes(buf[vtable_pos+field_offset_in_vtable:vtable_pos+field_offset_in_vtable+2], 'little')
    if field_rel == 0:
        return None
    return table_pos + field_rel


def _read_vector(buf, table_pos, field_id):
    """Return (data_start, count) for a FlatBuffers vector field."""
    field_pos = _vtable_field(buf, table_pos, field_id)
    if field_pos is None:
        return None, 0
    # The field value is an offset to the vector
    vec_offset = int.from_bytes(buf[field_pos:field_pos+4], 'little', signed=True)
    vec_pos = field_pos + vec_offset
    count = int.from_bytes(buf[vec_pos:vec_pos+4], 'little')
    return vec_pos + 4, count


def _read_string(buf, table_pos, field_id):
    field_pos = _vtable_field(buf, table_pos, field_id)
    if field_pos is None:
        return ''
    str_offset = int.from_bytes(buf[field_pos:field_pos+4], 'little', signed=True)
    str_pos = field_pos + str_offset
    str_len = int.from_bytes(buf[str_pos:str_pos+4], 'little')
    return buf[str_pos+4:str_pos+4+str_len].decode('utf-8', errors='replace')


def _read_scalar(buf, pos, size=4, signed=False):
    return int.from_bytes(buf[pos:pos+size], 'little', signed=signed)


def _deref_table(buf, ref_pos):
    """Dereference a table reference (indirect offset)."""
    offset = int.from_bytes(buf[ref_pos:ref_pos+4], 'little', signed=True)
    return ref_pos + offset


def _read_vec_elem_table(buf, vec_data_start, idx):
    """Get the absolute position of vector element idx (table reference)."""
    ref_pos = vec_data_start + idx * 4
    return _deref_table(buf, ref_pos)


def _read_vec_elem_i32(buf, vec_data_start, idx):
    pos = vec_data_start + idx * 4
    return _read_scalar(buf, pos, 4, signed=True)


def parse_tflite(path):
    with open(path, 'rb') as f:
        buf = bytearray(f.read())

    # Root table: offset stored at bytes 0-3 (offset from position 0)
    root_ref = _read_scalar(buf, 0, 4, signed=False)
    model = root_ref  # absolute position of Model table

    # TFLite Model schema:
    # field 0: version (uint32)
    # field 1: operator_codes ([OperatorCode])
    # field 2: subgraphs ([SubGraph])
    # field 3: description (string)
    # field 4: buffers ([Buffer])

    subgraphs_start, n_subgraphs = _read_vector(buf, model, 2)
    buffers_start, n_buffers = _read_vector(buf, model, 4)
    opcodes_start, n_opcodes = _read_vector(buf, model, 1)

    print(f"Subgraphs: {n_subgraphs}, Buffers: {n_buffers}, OpCodes: {n_opcodes}")

    # ── Helpers ────────────────────────────────────────────────────

    def get_buffer_data(buf_idx):
        buf_table = _read_vec_elem_table(buf, buffers_start, buf_idx)
        # Buffer: field 0 = data ([uint8])
        data_start, data_len = _read_vector(buf, buf_table, 0)
        if data_start is None or data_len == 0:
            return None
        return bytes(buf[data_start:data_start+data_len])

    def get_tensor(tensors_start, t_idx):
        t_table = _read_vec_elem_table(buf, tensors_start, t_idx)
        # Tensor: field 0=shape([int]), field 1=type(TensorType), field 2=buffer(uint), field 3=name
        shape_start, shape_len = _read_vector(buf, t_table, 0)
        shape = [_read_vec_elem_i32(buf, shape_start, i) for i in range(shape_len)] if shape_start else []
        dtype_pos = _vtable_field(buf, t_table, 1)
        dtype = _read_scalar(buf, dtype_pos, 1) if dtype_pos else 0
        buf_idx_pos = _vtable_field(buf, t_table, 2)
        buf_idx = _read_scalar(buf, buf_idx_pos, 4) if buf_idx_pos else 0
        name_str = _read_string(buf, t_table, 3)
        raw = get_buffer_data(buf_idx)
        if raw and dtype == 0:  # FLOAT32
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape) if shape else np.frombuffer(raw, dtype=np.float32)
        elif raw and dtype == 9:  # INT8
            arr = np.frombuffer(raw, dtype=np.int8).reshape(shape) if shape else np.frombuffer(raw, dtype=np.int8)
        else:
            arr = None
        return {'name': name_str, 'shape': shape, 'dtype': dtype, 'data': arr}

    def get_opcode(idx):
        oc_table = _read_vec_elem_table(buf, opcodes_start, idx)
        # OperatorCode: field 0=deprecated_builtin_code(int8), field 3=builtin_code(int32)
        # Try field 3 first (newer TFLite), fallback to field 0
        code_pos_new = _vtable_field(buf, oc_table, 3)
        if code_pos_new:
            return _read_scalar(buf, code_pos_new, 4, signed=True)
        code_pos_old = _vtable_field(buf, oc_table, 0)
        if code_pos_old:
            return _read_scalar(buf, code_pos_old, 1, signed=True) & 0xFF
        return 0

    # ── Parse first subgraph ───────────────────────────────────────

    sg_table = _read_vec_elem_table(buf, subgraphs_start, 0)
    # SubGraph: field 0=tensors, field 1=inputs, field 2=outputs, field 3=operators
    tensors_start, n_tensors = _read_vector(buf, sg_table, 0)
    ops_start, n_ops = _read_vector(buf, sg_table, 3)

    print(f"Tensors: {n_tensors}, Ops: {n_ops}")

    FULLY_CONNECTED = 9
    SOFTMAX = 25
    LOGISTIC = 14

    result_layers = []
    for op_i in range(n_ops):
        op_table = _read_vec_elem_table(buf, ops_start, op_i)
        # Operator: field 0=opcode_index(uint), field 1=inputs([int]), field 2=outputs([int])
        oc_idx_pos = _vtable_field(buf, op_table, 0)
        oc_idx = _read_scalar(buf, oc_idx_pos, 4) if oc_idx_pos else 0
        op_code = get_opcode(oc_idx)

        inp_start, n_inp = _read_vector(buf, op_table, 1)
        inputs = [_read_vec_elem_i32(buf, inp_start, i) for i in range(n_inp)] if inp_start else []

        print(f"  op {op_i}: code={op_code}, inputs={inputs}")

        if op_code == FULLY_CONNECTED and len(inputs) >= 2:
            weight_t = get_tensor(tensors_start, inputs[1])
            bias_t = get_tensor(tensors_start, inputs[2]) if len(inputs) > 2 else None
            if weight_t['data'] is not None:
                entry = {
                    'type': 'dense',
                    'weights': weight_t['data'].tolist(),
                    'shape': weight_t['shape'],
                }
                if bias_t and bias_t['data'] is not None:
                    entry['bias'] = bias_t['data'].tolist()
                    print(f"    weights shape={weight_t['shape']}, bias shape={bias_t['shape']}")
                result_layers.append(entry)
        elif op_code in (SOFTMAX, LOGISTIC):
            result_layers.append({'type': 'softmax' if op_code == SOFTMAX else 'sigmoid'})

    return result_layers


layers = parse_tflite(MODEL_PATH)
print(f"\nExtracted {len(layers)} layers:")
for i, l in enumerate(layers):
    t = l['type']
    if t == 'dense':
        print(f"  [{i}] dense  weights={l['shape']}")
    else:
        print(f"  [{i}] {t}")

with open(OUT_PATH, 'w') as f:
    json.dump(layers, f)
print(f"\nSaved → {OUT_PATH}")
