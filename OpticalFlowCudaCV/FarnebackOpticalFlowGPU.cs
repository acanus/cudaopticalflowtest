using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OpticalFlowCudaCV
{
    public class FarnebackOpticalFlowGPU
    {
        
        protected IntPtr ptr;
        public FarnebackOpticalFlowGPU(int numLevels=10,
                                        double pyrScale=0.5,
                                        bool fastPyramids=false,
                                        int winSize=20,
                                        int numIters=10,
                                        int polyN=5,
                                        double polySigma=0.8,
                                        int flags=0)
        {
            NativeMethods.CreateFarnebackOpticalFlow(out var ptrObj,  numLevels ,
                                         pyrScale,
                                         fastPyramids ,
                                         winSize ,
                                         numIters,
                                         polyN ,
                                         polySigma ,
                                         flags );
            NativeMethods.FarnebackOpticalFlowGet(ptrObj, out ptr);
        }
        public Mat Calc(IntPtr image1, IntPtr image2, int w, int h)
        {           
            NativeMethods.DenseOpticalFlowCalc(ptr, image1, image2, out var map_ptr, w, h);
            return new Mat(map_ptr);
        }
        public void Remap(Mat map,IntPtr image1,IntPtr image2, int w, int h)
        {
            NativeMethods.Remap(map.ptr, image1, image2, w, h);            
        }
    }
    public class OpticalFlowDual_TVL1:IDisposable
    {

        protected IntPtr ptr;
        public OpticalFlowDual_TVL1(double tau = 0.25,
                                    double lambda = 0.15,
                                    double theta = 0.3,
                                    int nscales = 5,
                                    int warps = 5,
                                    double epsilon = 0.01,
                                    int iterations = 300,
                                    double scaleStep = 0.8,
                                    double gamma = 0.0,
                                    bool useInitialFlow = false)
        {
            NativeMethods.CreateOpticalFlowDual_TVL1(out var ptrObj, 
                                                    tau,
                                                     lambda,
                                                     theta,
                                                     nscales,
                                                     warps,
                                                     epsilon,
                                                     iterations,
                                                     scaleStep,
                                                     gamma,
                                                     useInitialFlow);
            NativeMethods.OpticalFlowDual_TVL1Get(ptrObj, out ptr);
        }
        public Mat Calc(IntPtr image1, IntPtr image2, int w, int h)
        {
            NativeMethods.DenseOpticalFlowCalc(ptr, image1, image2, out var map_ptr, w, h);
            return new Mat(map_ptr);
        }

        public void Dispose()
        {
            
        }

        public void Remap(Mat map, IntPtr image1, IntPtr image2, int w, int h)
        {
            NativeMethods.Remap(map.ptr, image1, image2, w, h);
        }
    }
    public class OpticalFlowBrox : IDisposable
    {

        protected IntPtr ptr;
        public OpticalFlowBrox(double alpha = 0.197,
                                double gamma = 50.0,
                                double scale_factor = 0.8,
                                int inner_iterations = 5,
                                int outer_iterations = 150,
                                int solver_iterations = 10)
        {
            NativeMethods.CreateOpticalFlowBrox(out var ptrObj,
                                                    alpha,
                                                    gamma,
                                                    scale_factor,
                                                    inner_iterations,
                                                    outer_iterations,
                                                    solver_iterations);
            NativeMethods.OpticalFlowBroxGet(ptrObj, out ptr);
        }
        public Mat Calc(IntPtr image1, IntPtr image2, int w, int h)
        {
            NativeMethods.DenseOpticalFlowCalcFloat(ptr, image1, image2, out var map_ptr, w, h);
            return new Mat(map_ptr);
        }

        public void Dispose()
        {

        }

        public void Remap(Mat map, IntPtr image1, IntPtr image2, int w, int h)
        {
            NativeMethods.Remap(map.ptr, image1, image2, w, h);
        }
    }
    public class Mat
    {
        public IntPtr ptr;
        public Mat(IntPtr ptr)
        {
            this.ptr = ptr;
        }
    }

    public class ColorCorrection
    {
        public static int CalcColorCorectionMatrix(IntPtr image, int w, int h, IntPtr imageDraw, out double[] ccm)
        {

            int returnValue = NativeMethods.CalcColorCorectionMatrix(image, w, h, imageDraw, out var ccmPtr);
            if(returnValue == 1)
            {
                ccm = new double[9];
                Marshal.Copy(ccmPtr, ccm, 0, 9);
            }
            else
            {
                ccm = new double[9];
            }
            
            return returnValue;
        }
    }
}
