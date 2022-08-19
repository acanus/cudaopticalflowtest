using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpticalFlowCudaCV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpticalFlowCudaCV.Tests
{
    [TestClass()]
    public class FarnebackOpticalFlowGPUTests
    {
        [TestMethod()]
        public void CalcTest()
        {
            var obj = new FarnebackOpticalFlowGPU();
            //var result = obj.Calc(new byte[100 * 100], new byte[100 * 100], 100, 100);
            //obj.Remap(result, new byte[100 * 100], new byte[100 * 100], 100, 100);
        }
    }
}