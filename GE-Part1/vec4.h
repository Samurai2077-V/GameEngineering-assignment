#pragma once
#include <iostream>
//SIMD
#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h> 
//1.Force alignment through alignas(32) to suit SIMD operations and reduce memory access overhead.
//2.Use SSE to accelerate vector multiplication, addition, and subtraction operations to achieve data-level parallel processing.

// The `vec4` class represents a 4D vector and provides operations such as scaling, addition, subtraction, 
// normalization, and vector products (dot and cross).
class alignas(32) vec4 {         //Add alignas(32) before the class definition to ensure that the vec4 is aligned to 32 bytes.
public:
	union {
		struct {
			float x, y, z, w;     // Components of the vector
		};
		alignas(32) float v[4];   //Add alignas(32)       
	};

	// Constructor to initialize the vector with specified values.
	// Default values: x = 0, y = 0, z = 0, w = 1.
	// Input Variables:
	// - _x: X component of the vector
	// - _y: Y component of the vector
	// - _z: Z component of the vector
	// - _w: W component of the vector (default is 1.0)
	vec4(float _x = 0.f, float _y = 0.f, float _z = 0.f, float _w = 1.f)
		: x(_x), y(_y), z(_z), w(_w) {}
	// Displays the components of the vector in a readable format.
	void display() {
		std::cout << x << '\t' << y << '\t' << z << '\t' << w << std::endl;
	}

	// Scales the vector by a scalar value.
	// Input Variables:
	// - scalar: Value to scale the vector by
	// Returns a new scaled `vec4`.
	// 
	//Using SSE to implement operations
	vec4 operator*(float scalar) const {
		__m128 va = _mm_load_ps(v);
		__m128 vs = _mm_set1_ps(scalar);
		__m128 vc = _mm_mul_ps(va, vs);
		vec4 result;
		_mm_store_ps(result.v, vc);
		return result;
	}

	// Divides the vector by its W component and sets W to 1.
	// Useful for normalizing the W component after transformations.
	void divideW() {
		x /= w;
		y /= w;
		z /= w;
		w = 1.f;
	}

	// Accesses a vector component by index.
	// Input Variables:
	// - index: Index of the component (0 for x, 1 for y, 2 for z, 3 for w)
	// Returns a reference to the specified component.
	float& operator[](const unsigned int index) {
		return v[index];
	}
	// Accesses a vector component by index (const version).
	// Input Variables:
	// - index: Index of the component (0 for x, 1 for y, 2 for z, 3 for w)
	// Returns the specified component value.
	float operator[](const unsigned int index) const {
		return v[index];
	}

	// Subtracts another vector from this vector.
	// Input Variables:
	// - other: The vector to subtract
	// Returns a new `vec4` resulting from the subtraction.
	// 
	//Using SSE to implement operations
	vec4 operator-(const vec4& other) const {
		__m128 va = _mm_load_ps(v);
		__m128 vb = _mm_load_ps(other.v);
		__m128 vc = _mm_sub_ps(va, vb);
		vec4 result;
		_mm_store_ps(result.v, vc);
		return result;
	}

	// Adds another vector to this vector.
	// Input Variables:
	// - other: The vector to add
	// Returns a new `vec4` resulting from the addition.
	// 
	//Using SSE to implement operations
	vec4 operator+(const vec4& other) const {
		__m128 va = _mm_load_ps(v);        
		__m128 vb = _mm_load_ps(other.v); 
		__m128 vc = _mm_add_ps(va, vb);
		vec4 result;
		_mm_store_ps(result.v, vc);
		return result;
	}

	// Computes the cross product of two vectors.
	// Input Variables:
	// - v1: The first vector
	// - v2: The second vector
	// Returns a new `vec4` representing the cross product.
	static vec4 cross(const vec4& v1, const vec4& v2) {
		return vec4(
			v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x,
			0.0f // The W component is set to 0 for cross products
		);
	}
	// Computes the dot product of two vectors.
	// Input Variables:
	// - v1: The first vector
	// - v2: The second vector
	// Returns the dot product as a float.
	static float dot(const vec4& v1, const vec4& v2) {
		return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
	}
	// Normalizes the vector to make its length equal to 1.
	// This operation does not affect the W component.
	void normalise() {
		float length = std::sqrt(x * x + y * y + z * z);
		x /= length;
		y /= length;
		z /= length;
	}
};
