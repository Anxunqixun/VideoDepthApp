#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider selection and session recreate (mocked ORT; no CUDA required)."""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import engine


class FakeSession:
    def __init__(self, providers):
        self._providers = list(providers)

    def get_providers(self):
        return list(self._providers)


class FakeOrt:
    def __init__(self, available, fail_cuda=False):
        self.available = list(available)
        self.fail_cuda = fail_cuda
        self.created = []
        self.SessionOptions = lambda: SimpleNamespace(
            intra_op_num_threads=None,
            inter_op_num_threads=None,
            graph_optimization_level=None,
        )
        self.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL=99)

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, path, sess_options=None, providers=None):
        providers = list(providers or [])
        if self.fail_cuda and engine.CUDA_PROVIDER in providers:
            raise RuntimeError("CUDA EP init failed")
        self.created.append(providers)
        return FakeSession(providers)


class GpuSessionTests(unittest.TestCase):
    def setUp(self):
        engine.reset_session()

    def tearDown(self):
        engine.reset_session()

    def _load(self, ort, **kwargs):
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}):
            return engine.load_session(**kwargs)

    def test_cpu_only_providers(self):
        providers, note = engine._resolve_providers(
            FakeOrt(["CPUExecutionProvider"]), use_gpu=False
        )
        self.assertEqual(providers, [engine.CPU_PROVIDER])
        self.assertIsNone(note)

    def test_gpu_request_without_cuda_ep_falls_back(self):
        providers, note = engine._resolve_providers(
            FakeOrt(["CPUExecutionProvider"]), use_gpu=True
        )
        self.assertEqual(providers, [engine.CPU_PROVIDER])
        self.assertIn("已回退到 CPU", note)
        self.assertNotIn("TensorRT", note or "")
        self.assertNotIn("DML", note or "")
        self.assertNotIn("DirectML", note or "")

    def test_gpu_request_with_cuda_ep(self):
        providers, note = engine._resolve_providers(
            FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"]), use_gpu=True
        )
        self.assertEqual(providers, [engine.CUDA_PROVIDER, engine.CPU_PROVIDER])
        self.assertIsNone(note)

    def test_load_session_cpu(self):
        ort = FakeOrt(["CPUExecutionProvider"])
        self._load(ort, num_threads=2, use_gpu=False)
        info = engine.get_session_info()
        self.assertEqual(info["provider"], engine.CPU_PROVIDER)
        self.assertFalse(info["using_cuda"])
        self.assertFalse(info["use_gpu"])
        self.assertIsNone(info["note"])
        self.assertIn("CPU", engine.session_status_text())

    def test_load_session_gpu_fallback_status(self):
        ort = FakeOrt(["CPUExecutionProvider"])
        self._load(ort, num_threads=2, use_gpu=True)
        info = engine.get_session_info()
        self.assertEqual(info["provider"], engine.CPU_PROVIDER)
        self.assertTrue(info["use_gpu"])
        self.assertFalse(info["using_cuda"])
        self.assertIn("已回退到 CPU", info["note"])
        self.assertEqual(engine.session_status_text(), info["note"])
        self.assertEqual(ort.created, [[engine.CPU_PROVIDER]])

    def test_load_session_cuda_success(self):
        ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._load(ort, num_threads=2, use_gpu=True)
        info = engine.get_session_info()
        self.assertEqual(info["provider"], engine.CUDA_PROVIDER)
        self.assertTrue(info["using_cuda"])
        self.assertIsNone(info["note"])
        self.assertEqual(engine.session_status_text(), "当前推理设备：GPU（CUDA）")
        self.assertEqual(ort.created[-1], [engine.CUDA_PROVIDER, engine.CPU_PROVIDER])

    def test_cuda_init_failure_falls_back(self):
        ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"], fail_cuda=True)
        self._load(ort, num_threads=2, use_gpu=True)
        info = engine.get_session_info()
        self.assertEqual(info["provider"], engine.CPU_PROVIDER)
        self.assertIn("GPU 初始化失败", info["note"])
        self.assertEqual(ort.created[-1], [engine.CPU_PROVIDER])

    def test_recreate_on_gpu_toggle_or_thread_change(self):
        ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._load(ort, num_threads=2, use_gpu=False)
        self.assertEqual(len(ort.created), 1)
        self._load(ort, num_threads=2, use_gpu=False)
        self.assertEqual(len(ort.created), 1)
        self._load(ort, num_threads=2, use_gpu=True)
        self.assertEqual(len(ort.created), 2)
        self.assertEqual(engine.session_provider(), engine.CUDA_PROVIDER)
        self._load(ort, num_threads=4, use_gpu=True)
        self.assertEqual(len(ort.created), 3)
        self.assertEqual(engine.session_num_threads(), 4)

    def test_no_dml_or_tensorrt_in_provider_list(self):
        ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._load(ort, num_threads=1, use_gpu=True)
        for providers in ort.created:
            self.assertNotIn("DmlExecutionProvider", providers)
            self.assertNotIn("TensorrtExecutionProvider", providers)


if __name__ == "__main__":
    unittest.main()
