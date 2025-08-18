#ifndef SE3_H
#define SE3_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

// SO3 implementation
template<typename Scalar>
class SO3 {
public:
    using Quaternion = Eigen::Quaternion<Scalar>;
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

private:
    Quaternion rotation_;

public:
    SO3() : rotation_(Quaternion::Identity()) {}
    SO3(const Quaternion& q) : rotation_(q.normalized()) {}
    SO3(const Matrix3& R) : rotation_(R) {}

    static SO3 Identity() { return SO3(); }
    
    static SO3 exp(const Vector3& omega) {
        Scalar theta = omega.norm();
        if (theta < 1e-8) {
            return SO3();
        }
        Vector3 axis = omega / theta;
        return SO3(Quaternion(Eigen::AngleAxis<Scalar>(theta, axis)));
    }

    // Hat operator: converts vector to skew-symmetric matrix
    static Matrix3 hat(const Vector3& v) {
        Matrix3 S;
        S << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return S;
    }

    Matrix3 matrix() const { return rotation_.toRotationMatrix(); }
    const Quaternion& unit_quaternion() const { return rotation_; }

    // Log map: converts rotation to axis-angle vector
    Vector3 log() const {
        Eigen::AngleAxis<Scalar> angleAxis(rotation_);
        return angleAxis.angle() * angleAxis.axis();
    }

    SO3 inverse() const { return SO3(rotation_.inverse()); }
    
    SO3 operator*(const SO3& other) const {
        return SO3(rotation_ * other.rotation_);
    }

    Vector3 operator*(const Vector3& v) const {
        return rotation_ * v;
    }
};

using SO3f = SO3<float>;
using SO3d = SO3<double>;

// Sim3 implementation (Similarity transformation: SE3 + scale)
template<typename Scalar>
class Sim3 {
public:
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    using Quaternion = Eigen::Quaternion<Scalar>;

private:
    SO3<Scalar> rotation_;
    Vector3 translation_;
    Scalar scale_;

public:
    // Constructors
    Sim3() : rotation_(SO3<Scalar>::Identity()), translation_(Vector3::Zero()), scale_(1.0) {}
    
    Sim3(const SO3<Scalar>& R, const Vector3& t, Scalar s) 
        : rotation_(R), translation_(t), scale_(s) {}
        
    Sim3(const Matrix3& R, const Vector3& t, Scalar s) 
        : rotation_(R), translation_(t), scale_(s) {}

    // Static constructors
    static Sim3 Identity() {
        return Sim3();
    }

    // Accessors
    const SO3<Scalar>& so3() const { return rotation_; }
    const Vector3& translation() const { return translation_; }
    Scalar scale() const { return scale_; }
    
    Matrix3 rotationMatrix() const { return rotation_.matrix(); }
    
    Matrix4 matrix() const {
        Matrix4 T = Matrix4::Identity();
        T.template block<3,3>(0,0) = scale_ * rotationMatrix();
        T.template block<3,1>(0,3) = translation_;
        return T;
    }

    // Operations
    Sim3 inverse() const {
        SO3<Scalar> R_inv = rotation_.inverse();
        Scalar s_inv = 1.0 / scale_;
        return Sim3(R_inv, -s_inv * (R_inv * translation_), s_inv);
    }

    Sim3 operator*(const Sim3& other) const {
        return Sim3(rotation_ * other.rotation_, 
                   scale_ * (rotation_ * other.translation_) + translation_,
                   scale_ * other.scale_);
    }

    Vector3 operator*(const Vector3& point) const {
        return scale_ * (rotation_ * point) + translation_;
    }

    // Setters
    void setRotation(const SO3<Scalar>& R) { rotation_ = R; }
    void setTranslation(const Vector3& t) { translation_ = t; }
    void setScale(Scalar s) { scale_ = s; }
};

using Sim3f = Sim3<float>;
using Sim3d = Sim3<double>;

/**
 * @brief SE3 group implementation using Eigen
 * Replaces SE3f with Eigen-based implementation
 */
template<typename Scalar>
class SE3 {
public:
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vector6 = Eigen::Matrix<Scalar, 6, 1>;
    using Quaternion = Eigen::Quaternion<Scalar>;

private:
    Quaternion rotation_;
    Vector3 translation_;

public:
    // Constructors
    SE3() : rotation_(Quaternion::Identity()), translation_(Vector3::Zero()) {}
    
    SE3(const Quaternion& q, const Vector3& t) : rotation_(q.normalized()), translation_(t) {}
    
    SE3(const Matrix3& R, const Vector3& t) : rotation_(R), translation_(t) {}
    
    SE3(const Matrix4& T) {
        rotation_ = Quaternion(T.template block<3,3>(0,0));
        translation_ = T.template block<3,1>(0,3);
    }

    // Static constructors
    static SE3 Identity() {
        return SE3();
    }

    static SE3 exp(const Vector6& xi) {
        Vector3 rho = xi.template head<3>();
        Vector3 phi = xi.template tail<3>();
        
        Scalar phi_norm = phi.norm();
        Matrix3 V;
        
        if (phi_norm < 1e-8) {
            V = Matrix3::Identity();
        } else {
            Matrix3 Omega = SO3<Scalar>::hat(phi);
            V = Matrix3::Identity() + 
                (1 - cos(phi_norm)) / (phi_norm * phi_norm) * Omega +
                (phi_norm - sin(phi_norm)) / (phi_norm * phi_norm * phi_norm) * Omega * Omega;
        }
        
        return SE3(Quaternion(Eigen::AngleAxis<Scalar>(phi_norm, phi.normalized())), V * rho);
    }

    // Accessors
    const Quaternion& unit_quaternion() const { return rotation_; }
    const Vector3& translation() const { return translation_; }
    Vector3& translation() { return translation_; }  // Non-const version
    
    SO3<Scalar> so3() const { return SO3<Scalar>(rotation_); }  // Get SO3 part
    
    Matrix3 rotationMatrix() const { return rotation_.toRotationMatrix(); }
    
    Matrix4 matrix() const {
        Matrix4 T = Matrix4::Identity();
        T.template block<3,3>(0,0) = rotationMatrix();
        T.template block<3,1>(0,3) = translation_;
        return T;
    }

    Matrix4 matrix4x4() const { return matrix(); }
    
    Eigen::Matrix<Scalar, 3, 4> matrix3x4() const {
        Eigen::Matrix<Scalar, 3, 4> T;
        T.template block<3,3>(0,0) = rotationMatrix();
        T.template block<3,1>(0,3) = translation_;
        return T;
    }

    // Operations
    SE3 inverse() const {
        Quaternion q_inv = rotation_.inverse();
        return SE3(q_inv, -(q_inv * translation_));
    }

    SE3 operator*(const SE3& other) const {
        return SE3(rotation_ * other.rotation_, 
                   rotation_ * other.translation_ + translation_);
    }

    Vector3 operator*(const Vector3& point) const {
        return rotation_ * point + translation_;
    }

    // Setters
    void setQuaternion(const Quaternion& q) {
        rotation_ = q.normalized();
    }

    void setTranslation(const Vector3& t) {
        translation_ = t;
    }

    void setRotationMatrix(const Matrix3& R) {
        rotation_ = Quaternion(R);
    }

    // Type casting
    template<typename TargetScalar>
    SE3<TargetScalar> cast() const {
        return SE3<TargetScalar>(
            rotation_.template cast<TargetScalar>(),
            translation_.template cast<TargetScalar>()
        );
    }

    // Logarithm map
    Vector6 log() const {
        Vector6 xi;
        Vector3 phi = logSO3(rotation_);
        Scalar phi_norm = phi.norm();
        
        Matrix3 V_inv;
        if (phi_norm < 1e-8) {
            V_inv = Matrix3::Identity();
        } else {
            Matrix3 Omega = SO3<Scalar>::hat(phi);
            V_inv = Matrix3::Identity() - 0.5 * Omega +
                    (2 * sin(phi_norm) - phi_norm * (1 + cos(phi_norm))) /
                    (2 * phi_norm * phi_norm * sin(phi_norm)) * Omega * Omega;
        }
        
        xi.template head<3>() = V_inv * translation_;
        xi.template tail<3>() = phi;
        return xi;
    }

private:
    // Helper functions
    static Vector3 logSO3(const Quaternion& q) {
        Scalar w = q.w();
        Vector3 v = q.vec();
        Scalar v_norm = v.norm();
        
        if (v_norm < 1e-8) {
            return 2 * v;
        }
        
        Scalar theta = 2 * atan2(v_norm, abs(w));
        if (w < 0) theta = 2 * M_PI - theta;
        
        return theta / v_norm * v;
    }
};

// Type aliases
using SE3f = SE3<float>;
using SE3d = SE3<double>;

#endif // SE3_H
