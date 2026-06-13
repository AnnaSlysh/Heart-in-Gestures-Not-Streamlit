"""
Extract LSTM + Dense weights from dynamic_classifier.tflite.
Outputs dynamic_model_weights.json for pure-JS inference.

TFLite LSTM (UNIDIRECTIONAL_SEQUENCE_LSTM) tensor layout (standard 20-input form):
  0  input
  1  input_to_input_weights    (units, input_size)   i-gate
  2  input_to_forget_weights   (units, input_size)   f-gate
  3  input_to_cell_weights     (units, input_size)   g-gate
  4  input_to_output_weights   (units, input_size)   o-gate
  5  recurrent_to_input_weights  (units, units)
  6  recurrent_to_forget_weights (units, units)
  7  recurrent_to_cell_weights   (units, units)
  8  recurrent_to_output_weights (units, units)
  9  cell_to_input_weights     (optional peephole, skip)
  10 cell_to_forget_weights    (optional peephole, skip)
  11 cell_to_output_weights    (optional peephole, skip)
  12 input_gate_bias           (units,)
  13 forget_gate_bias          (units,)
  14 cell_gate_bias            (units,)
  15 output_gate_bias          (units,)
  16 projection_weights        (optional, skip)
  17 projection_bias           (optional, skip)
  18 output_state_in
  19 cell_state_in
"""
import json, numpy as np, os, struct

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'dynamic_classifier', 'dynamic_classifier.tflite')
OUT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'dynamic_classifier', 'dynamic_model_weights.json')


def _vtable_field(buf, table_pos, field_id):
    soff = int.from_bytes(buf[table_pos:table_pos+4], 'little', signed=True)
    vt = table_pos - soff
    vt_size = int.from_bytes(buf[vt:vt+2], 'little')
    fp = 4 + field_id * 2
    if fp >= vt_size: return None
    rel = int.from_bytes(buf[vt+fp:vt+fp+2], 'little')
    return (table_pos + rel) if rel else None


def _read_vector(buf, table_pos, field_id):
    fp = _vtable_field(buf, table_pos, field_id)
    if fp is None: return None, 0
    off = int.from_bytes(buf[fp:fp+4], 'little', signed=True)
    vec = fp + off
    count = int.from_bytes(buf[vec:vec+4], 'little')
    return vec + 4, count


def _read_scalar(buf, pos, size=4, signed=False):
    return int.from_bytes(buf[pos:pos+size], 'little', signed=signed)


def _deref(buf, ref_pos):
    off = int.from_bytes(buf[ref_pos:ref_pos+4], 'little', signed=True)
    return ref_pos + off


def _vec_table(buf, vec_start, idx):
    rp = vec_start + idx * 4
    return _deref(buf, rp)


def _vec_i32(buf, vec_start, idx):
    return _read_scalar(buf, vec_start + idx * 4, 4, signed=True)


def parse(path):
    with open(path, 'rb') as f:
        buf = bytearray(f.read())

    root = _read_scalar(buf, 0, 4)
    # Model fields: 0=version, 1=operator_codes, 2=subgraphs, 4=buffers
    subgraphs_s, _ = _read_vector(buf, root, 2)
    buffers_s,   _ = _read_vector(buf, root, 4)
    opcodes_s,   _ = _read_vector(buf, root, 1)

    sg = _vec_table(buf, subgraphs_s, 0)
    tensors_s,  n_t = _read_vector(buf, sg, 0)
    ops_s,      n_o = _read_vector(buf, sg, 3)

    def buf_data(buf_idx):
        bt = _vec_table(buf, buffers_s, buf_idx)
        ds, dl = _read_vector(buf, bt, 0)
        return bytes(buf[ds:ds+dl]) if ds and dl else None

    def tensor(idx):
        t = _vec_table(buf, tensors_s, idx)
        shape_s, sl = _read_vector(buf, t, 0)
        shape  = [_vec_i32(buf, shape_s, i) for i in range(sl)] if shape_s else []
        dtype  = _read_scalar(buf, _vtable_field(buf, t, 1) or 0, 1) if _vtable_field(buf, t, 1) else 0
        bi     = _read_scalar(buf, _vtable_field(buf, t, 2), 4) if _vtable_field(buf, t, 2) else 0
        raw    = buf_data(bi)
        if raw and dtype == 0:  # float32
            arr = np.frombuffer(raw, np.float32).reshape(shape) if shape else np.frombuffer(raw, np.float32)
        elif raw and dtype == 9:  # int8
            arr = np.frombuffer(raw, np.int8).reshape(shape) if shape else np.frombuffer(raw, np.int8)
        else:
            arr = None
        # name
        fp = _vtable_field(buf, t, 3)
        if fp:
            sr = _read_scalar(buf, fp, 4, signed=True)
            sp = fp + sr
            sl2 = _read_scalar(buf, sp, 4)
            name = buf[sp+4:sp+4+sl2].decode('utf-8', errors='replace')
        else:
            name = ''
        return name, shape, dtype, arr

    def opcode(idx):
        ot = _vec_table(buf, opcodes_s, idx)
        fp3 = _vtable_field(buf, ot, 3)
        if fp3: return _read_scalar(buf, fp3, 4, signed=True)
        fp0 = _vtable_field(buf, ot, 0)
        return (_read_scalar(buf, fp0, 1, signed=True) & 0xFF) if fp0 else 0

    # Operator codes:
    # 35 = LSTM (basic), 44 = UNIDIRECTIONAL_SEQUENCE_LSTM, 9 = FULLY_CONNECTED, 25 = SOFTMAX, 14 = LOGISTIC
    LSTM_OPS   = {35, 44}
    FC_OP      = 9
    SOFTMAX_OP = 25
    RELU_OP    = 19

    layers = []
    for op_i in range(n_o):
        op = _vec_table(buf, ops_s, op_i)
        oci = _read_scalar(buf, _vtable_field(buf, op, 0), 4) if _vtable_field(buf, op, 0) else 0
        code = opcode(oci)
        inp_s, n_inp = _read_vector(buf, op, 1)
        inputs = [_vec_i32(buf, inp_s, i) for i in range(n_inp)] if inp_s else []
        out_s, n_out = _read_vector(buf, op, 2)
        outputs = [_vec_i32(buf, out_s, i) for i in range(n_out)] if out_s else []
        print(f"  op {op_i}: code={code} inputs={inputs[:6]}{'...' if len(inputs)>6 else ''}")

        if code in LSTM_OPS:
            # Standard TFLite LSTM inputs: see module docstring
            # indices 1-4: input weights (ifgo), 5-8: recurrent weights (ifgo)
            # indices 12-15: biases (ifgo)
            def w(idx):
                if idx >= len(inputs) or inputs[idx] < 0: return None
                _, shape, _, arr = tensor(inputs[idx])
                return arr

            lstm_data = {
                'type': 'lstm',
                'return_sequences': True,  # assume return_seq for all but last
                'Wi': w(1).tolist() if w(1) is not None else None,  # input_to_i
                'Wf': w(2).tolist() if w(2) is not None else None,  # input_to_f
                'Wg': w(3).tolist() if w(3) is not None else None,  # input_to_g
                'Wo': w(4).tolist() if w(4) is not None else None,  # input_to_o
                'Ui': w(5).tolist() if w(5) is not None else None,  # recurrent_to_i
                'Uf': w(6).tolist() if w(6) is not None else None,  # recurrent_to_f
                'Ug': w(7).tolist() if w(7) is not None else None,  # recurrent_to_g
                'Uo': w(8).tolist() if w(8) is not None else None,  # recurrent_to_o
                'bi': w(12).tolist() if w(12) is not None else None,
                'bf': w(13).tolist() if w(13) is not None else None,
                'bg': w(14).tolist() if w(14) is not None else None,
                'bo': w(15).tolist() if w(15) is not None else None,
            }
            if lstm_data['Wi'] is not None:
                print(f"    LSTM units={len(lstm_data['Wi'])}, input_size={len(lstm_data['Wi'][0])}")
            layers.append(lstm_data)

        elif code == FC_OP and len(inputs) >= 2 and inputs[1] >= 0:
            _, ws, _, warr = tensor(inputs[1])
            _, bs, _, barr = tensor(inputs[2]) if len(inputs) > 2 and inputs[2] >= 0 else (None, None, None, None)
            if warr is not None:
                print(f"    Dense shape={ws}")
                layers.append({'type': 'dense', 'weights': warr.tolist(), 'shape': ws, 'bias': barr.tolist() if barr is not None else None})

        elif code == SOFTMAX_OP:
            layers.append({'type': 'softmax'})
        elif code == RELU_OP:
            layers.append({'type': 'relu'})

    return layers


layers = parse(MODEL_PATH)
print(f"\nExtracted {len(layers)} layers")
# mark which LSTM returns sequences vs not
lstm_count = sum(1 for l in layers if l.get('type') == 'lstm')
lstm_seen = 0
for l in layers:
    if l.get('type') == 'lstm':
        lstm_seen += 1
        l['return_sequences'] = lstm_seen < lstm_count

with open(OUT_PATH, 'w') as f:
    json.dump(layers, f)
print(f"Saved to {OUT_PATH} ({os.path.getsize(OUT_PATH)//1024} KB)")
