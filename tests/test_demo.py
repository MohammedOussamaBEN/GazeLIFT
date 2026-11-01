from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hf-space'))
import inference

class DemoTests(unittest.TestCase):
    def test_gaze_center_and_constant_duration(self):
        h=inference.prepare_gaze([[100,50,4]],(201,101),32)
        self.assertEqual(tuple(h.shape),(1,1,512,512))
        y,x=np.unravel_index(h[0,0].argmax().item(),(512,512))
        self.assertLessEqual(abs(x-255.5),1);self.assertLessEqual(abs(y-255.5),1)
        torch.testing.assert_close(h,inference.prepare_gaze([[100,50,999]],(201,101),32))

    def test_invalid_gaze(self):
        for rows in [[],[[300,10,1]],[[2,3,-1]],[[float('nan'),3,1]]]:
            with self.assertRaises(ValueError):inference.prepare_gaze(rows,(201,101),32)

    def test_csv_matching_order_and_units(self):
        with tempfile.TemporaryDirectory() as f:
            p=Path(f)/'fix.csv';p.write_text('IMAGE,CURRENT_FIX_INDEX,CURRENT_FIX_X,CURRENT_FIX_Y,CURRENT_FIX_DURATION\na.png,2,1,1,3\nb.png,1,0,0,2\na.png,1,0.5,0.5,1\n')
            np.testing.assert_allclose(inference.read_fixations(p,'a.png',(201,101),'Normalized [0,1]'),[[100,50,1],[200,100,3]])

    def test_missing_checkpoint_stops_inference(self):
        with self.assertRaises(FileNotFoundError):inference.verified_checkpoint('kvasir')

    def test_architecture_forward(self):
        torch.set_num_threads(2);m=inference.TransposeCNNGazePyramidDecoder().eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()),817993)
        with torch.inference_mode():out=m(torch.zeros(1,3,32,32),[torch.zeros(1,1024,2,2) for _ in range(4)],torch.ones(1,1,32,32))
        self.assertEqual(tuple(out.shape),(1,1,32,32));self.assertTrue(torch.isfinite(out).all())

if __name__=='__main__':unittest.main()
