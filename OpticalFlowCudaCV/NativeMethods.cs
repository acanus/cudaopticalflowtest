using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OpticalFlowCudaCV
{
    public static partial class NativeMethods
    {
        public const string DllExtern = "cudaopticalflow";
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void CreateFarnebackOpticalFlow(out IntPtr returnValue, int numLevels,
                                                                                    double pyrScale,
                                                                                    bool fastPyramids,
                                                                                    int winSize,
                                                                                    int numIters,
                                                                                    int polyN,
                                                                                    double polySigma,
                                                                                    int flags);        
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void FarnebackOpticalFlowGet(IntPtr ptr, out IntPtr returnValue);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void CreateOpticalFlowDual_TVL1(out IntPtr returnValue, 
                                                                    double tau ,
                                                                    double lambda ,
                                                                    double theta ,
                                                                    int nscales ,
                                                                    int warps ,
                                                                    double epsilon ,
                                                                    int iterations ,
                                                                    double scaleStep ,
                                                                    double gamma ,
                                                                    bool useInitialFlow );
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void OpticalFlowDual_TVL1Get(IntPtr ptr, out IntPtr returnValue);


        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void CreateOpticalFlowBrox(out IntPtr returnValue,
                                                                    double alpha,
                                                                    double gamma,
                                                                    double scale_factor,
                                                                    int inner_iterations,
                                                                    int outer_iterations ,
                                                                    int solver_iterations);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void OpticalFlowBroxGet(IntPtr ptr, out IntPtr returnValue);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void DenseOpticalFlowCalc(IntPtr obj, IntPtr image1, IntPtr image2, out IntPtr map_vector, int w, int h);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void DenseOpticalFlowCalcFloat(IntPtr obj, IntPtr image1, IntPtr image2, out IntPtr map_vector, int w, int h);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void Remap(IntPtr map, IntPtr image,IntPtr imageMapped,int w,int h);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int CalcColorCorectionMatrix(IntPtr image, int w, int h, IntPtr imageDraw, out IntPtr ccm);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int ApplyCCM(IntPtr image, int w, int h, double[] ccm, IntPtr out_img);
    }

}
