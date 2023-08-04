using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpticalFlowCudaCV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace OpticalFlowCudaCV.Tests
{
    [TestClass()]
    public class FarnebackOpticalFlowGPUTests
    {
        [TestMethod()]
        public void ColorCorrectionTest()
        {
            OpenCvSharp.Mat image = new OpenCvSharp.Mat("D:\\images\\colorChecker\\7.bmp");
            var imageDraw = image.Clone();
            var imageOut = image.Clone();
            ColorCorrection.CalcColorCorectionMatrix(image.Data,  image.Width, image.Height, imageDraw.Data, out double[] result);
            ColorCorrection.ApplyCCM(image.Data, image.Width, image.Height,result, imageOut.Data);
            Cv2.ImShow("aadas", imageOut);
            Cv2.WaitKey(-1);
            Console.WriteLine(result);
        }
    }
}