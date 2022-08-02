#include <Python.h>
#include <stella_vslam/system.h>
#include <stella_vslam/config.h>
#include <stella_vslam/type.h>
#include <stella_vslam/publish/map_publisher.h>
#include <stella_vslam/publish/frame_publisher.h>
#include <pangolin_viewer/viewer.h>
#include "pybind11/stl.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
// #include "nlohmann/json_fwd.hpp"
#include "yaml-cpp/yaml.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <thread>

namespace py = pybind11;

class NDArrayConverter {
public:
    static bool init_numpy();   // must call this first, or the other routines don't work!
    static bool toMat(PyObject* o, cv::Mat &m);
    static PyObject* toNDArray(const cv::Mat& mat);
};

namespace pybind11{namespace detail{
template <> struct type_caster<cv::Mat>{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));
    bool load(handle src, bool){
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval){
        return handle(NDArrayConverter::toNDArray(m));
    }
};    
}} // namespace pybind11::detail

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#if PY_VERSION_HEX >= 0x03000000
    #define PyInt_Check PyLong_Check
    #define PyInt_AsLong PyLong_AsLong
#endif

struct Tmp {
    const char * name;
    Tmp(const char * name):name(name){}
} info("return value");

bool NDArrayConverter::init_numpy(){
    // this has to be in this file, since PyArray_API is defined as static
    import_array1(false);
    return true;
}

/*
 * The following conversion functions are taken/adapted from OpenCV's cv2.cpp file
 * inside modules/python/src2 folder (OpenCV 3.1.0)
 */
static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...){
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads(){
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL(){
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try {PyAllowThreads allowThreads; expr;} \
catch (const cv::Exception &e){PyErr_SetString(opencv_error, e.what()); return 0;}

using namespace cv;

class NumpyAllocator : public MatAllocator {
public:
    NumpyAllocator(){stdAllocator = Mat::getStdAllocator();}
    ~NumpyAllocator(){}

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const{
        UMatData* u = new UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( int i = 0; i < dims - 1; i++ )
            step[i] = (size_t)_strides[i];
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

#if CV_MAJOR_VERSION < 4
    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const
#else
    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const
#endif
    {
        if( data != 0 ){
            CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
            _sizes[dims++] = cn;
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        return allocate(o, dims0, sizes, type, step);
    }

#if CV_MAJOR_VERSION < 4
    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const
#else
    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const
#endif
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const{
        if(!u)
            return;
        PyEnsureGIL gil;
        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);
        if(u->refcount == 0){
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};

NumpyAllocator g_numpyAllocator;

bool NDArrayConverter::toMat(PyObject *o, Mat &m){
    bool allowND = true;
    if(!o || o == Py_None){
        if(!m.data)
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if(PyInt_Check(o)){
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if(PyFloat_Check(o)){
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if(PyTuple_Check(o)){
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = Mat(sz, 1, CV_64F);
        for(i = 0; i < sz; i++){
            PyObject* oi = PyTuple_GET_ITEM(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else{
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    if(!PyArray_Check(o)){
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if(type < 0){
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG ){
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        } else {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM){
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for(int i = ndims-1; i >= 0 && !needcopy; i--)
        if(
            (i == ndims-1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize)     ||
            (i  < ndims-1 && _sizes[i] > 1 &&         _strides[i] < _strides[i+1]) 
        )
            needcopy = true;

    if(ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2])
        needcopy = true;

    if(needcopy){
        if(needcast){
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        } else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }
        _strides = PyArray_STRIDES(oarr);
    }

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for(int i = ndims - 1; i >= 0; --i){
        size[i] = (int)_sizes[i];
        if ( size[i] > 1 ){
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        } else {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // handle degenerate case
    if(ndims == 0){
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if(ismultichannel){
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if(ndims > 2 && !allowND){
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    m.addref();

    if(!needcopy){
        Py_INCREF(o);
    }
    m.allocator = &g_numpyAllocator;

    return true;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m){
    if( !m.data ){
        Py_RETURN_NONE;
    }
    Mat temp, *p = (Mat*)&m;
    if(!p->u || p->allocator != &g_numpyAllocator){
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}

/*
 * Receives a pointer to pose matrix.
 * Returns a pair with (retval, pose)
 * retval: true if ok, false if the pose is invalid (in this case pose and position will have uninitialized values)
 * pose: Mat44x, automatically 4x4 float numpy ndarray
 * 
 * To get the 3 element position (x,y,z) in Python use pose[0:2, 3]
 */
std::pair<bool, stella_vslam::Mat44_t> ptr2pose(std::shared_ptr<stella_vslam::Mat44_t> pose_ptr){
    if(pose_ptr == nullptr)
        // No pose
        return std::pair<bool, stella_vslam::Mat44_t>(false, stella_vslam::Mat44_t());
    else
        // Pose ok
        return std::pair<bool, stella_vslam::Mat44_t>(true, (stella_vslam::Mat44_t) *pose_ptr);
}

/*
 * Python bindings, module stella_vslam
 * Minimum requirement for stella_vslam operation: some functions in classes system and config.
 * Classes, functions and arguments keep their original stella_vslam names.
 */
using namespace stella_vslam;
PYBIND11_MODULE(stella_vslam, m){
    NDArrayConverter::init_numpy();

    py::class_<stella_vslam::camera::image_bounds>(m, "camera::image_bounds")
    .def_readonly("min_x_", &stella_vslam::camera::image_bounds::min_x_)
    .def_readonly("max_x_", &stella_vslam::camera::image_bounds::max_x_)
    .def_readonly("min_y_", &stella_vslam::camera::image_bounds::min_y_)
    .def_readonly("max_y_", &stella_vslam::camera::image_bounds::max_y_)
    ;

    py::class_<stella_vslam::camera::base, std::shared_ptr<stella_vslam::camera::base>>(m, "camera::base")
    .def_readonly("img_bounds_", &stella_vslam::camera::base::img_bounds_)    
    ;

    py::class_<config, std::shared_ptr<config>>(m, "config")
        .def(py::init<const std::string&>(), py::arg("config_file_path"))
        .def(py::init<const YAML::Node&, const std::string&>(), py::arg("yaml_node"), py::arg("config_file_path") = "")
        .def_readonly("yaml_node_", &config::yaml_node_)
        .def_readonly("camera_", &config::camera_);

    py::class_<stella_vslam::system>(m, "system")
        // Init & finish
        .def(py::init<const std::shared_ptr<config>&, const std::string&>(), py::arg("cfg"), py::arg("vocab_file_path"))
        .def("startup", &system::startup, py::arg("need_initialize") = true)
        .def("shutdown", &system::shutdown)
        
        .def("feed_monocular_frame", [](stella_vslam::system &self, const cv::Mat &img, const double timestamp, const cv::Mat& mask = cv::Mat{}) {
                py::gil_scoped_release release; 
                std::shared_ptr<Mat44_t> matrix = self.feed_monocular_frame(img, timestamp, mask); 
                py::gil_scoped_acquire acquire;
                return ptr2pose(matrix);
            }, 
            py::arg("img"), py::arg("timestamp"), py::arg("mask") = cv::Mat{})
        
        .def("feed_stereo_frame", [](stella_vslam::system &self, const cv::Mat &left_img, const cv::Mat &right_img, const double timestamp, const cv::Mat& mask = cv::Mat{}) { 
                py::gil_scoped_release release;
                std::shared_ptr<Mat44_t> matrix = self.feed_stereo_frame(left_img, right_img, timestamp, mask); 
                py::gil_scoped_acquire acquire;
                return ptr2pose(matrix);
            }, 
            py::arg("left_img"), py::arg("right_img"), py::arg("timestamp"), py::arg("mask") = cv::Mat{})
        
        .def("feed_RGBD_frame", [](stella_vslam::system &self, const cv::Mat &rgb_img, const cv::Mat &depthmap, const double timestamp, const cv::Mat& mask = cv::Mat{}) { 
                py::gil_scoped_release release;
                std::shared_ptr<Mat44_t> matrix = self.feed_RGBD_frame(rgb_img, depthmap, timestamp, mask); 
                py::gil_scoped_acquire acquire;
                return ptr2pose(matrix);
            }, 
            py::arg("rgb_img"), py::arg("depthmap"), py::arg("timestamp"), py::arg("mask") = cv::Mat{})

        // Map save & load
        .def("load_map_database", &system::load_map_database, py::arg("path"))
        .def("save_map_database", &system::save_map_database, py::arg("path"))

        .def("save_frame_trajectory", &system::save_frame_trajectory, py::arg("path"), py::arg("format"))
        .def("save_keyframe_trajectory", &system::save_keyframe_trajectory, py::arg("path"), py::arg("format"))

        // Frame and map publishers
        .def("get_map_publisher", &system::get_map_publisher)
        .def("get_frame_publisher", &system::get_frame_publisher)

        // System controls
        .def("mapping_module_is_enabled", &system::mapping_module_is_enabled)
        .def("enable_mapping_module", &system::enable_mapping_module)
        .def("disable_mapping_module", &system::disable_mapping_module)

        .def("loop_detector_is_enabled", &system::loop_detector_is_enabled)
        .def("enable_loop_detector", &system::enable_loop_detector)
        .def("disable_loop_detector", &system::disable_loop_detector)

        .def("loop_BA_is_running", &system::loop_BA_is_running)
        .def("abort_loop_BA", &system::abort_loop_BA)
        
        .def("tracker_is_paused", &system::tracker_is_paused)
        .def("pause_tracker", &system::pause_tracker)
        .def("resume_tracker", &system::resume_tracker)
        
        .def("reset_is_requested", &system::reset_is_requested)
        .def("request_reset", &system::request_reset)

        .def("terminate_is_requested", &system::terminate_is_requested)
        .def("request_terminate", &system::request_terminate)
        ;

    py::class_<pangolin_viewer::viewer>(m, "viewer")
        .def(py::init([](const YAML::Node& yaml_node_, stella_vslam::system* SLAM)  
        {  
            return new pangolin_viewer::viewer(yaml_node_, SLAM, SLAM->get_frame_publisher(), SLAM->get_map_publisher());
        }))
        
        // https://stackoverflow.com/questions/60410178/how-to-invoke-python-function-as-a-callback-inside-c-thread-using-pybind11
        // Python GIL pervents us from parallelizing SLAM and the viewer using threads. We allow parallelization by adding a call guard
        .def("run", &pangolin_viewer::viewer::run, py::call_guard<py::gil_scoped_release>())
        .def("request_terminate", &pangolin_viewer::viewer::request_terminate)

        // This is not a function present in the original stella_vslam system
        // .def("take_screenshot", &pangolin_viewer::viewer::take_screenshot, py::arg("filename"))
        .def("take_screenshot", &pangolin_viewer::viewer::take_screenshot, py::arg("filename"))
        
        // Not recommended, but useful to test stuff and avoid the GIL
        .def("run_in_detached_thread", [](pangolin_viewer::viewer &self){                
                std::thread thread([&]() {
                    std::cout << "Running viewer" << "\n";
                    self.run();
                });
                thread.detach();                
        }, py::call_guard<py::gil_scoped_release>())
        ;

    py::class_<YAML::Node>(m, "YamlNode")
        .def(py::init<const std::string &>())
        .def("__getitem__",
            [](const YAML::Node node, const std::string& key){
              return node[key];
            })
        .def("__iter__",
            [](const YAML::Node &node) {
              return py::make_iterator(node.begin(), node.end());},
             py::keep_alive<0, 1>())
        .def("__str__",
             [](const YAML::Node& node) {
               YAML::Emitter out;
               out << node;
               return std::string(out.c_str());
             })
        .def("type", &YAML::Node::Type)
        .def("__len__", &YAML::Node::size)
        ;      
}