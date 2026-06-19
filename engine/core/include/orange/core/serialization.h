#pragma once

#define NOMINMAX

#undef min
#undef max

#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <mutex>
#include <cstring>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// Multi-format mesh / point-cloud serialization (XYZ, OFF, STL, OBJ, PLY, PTS,
// ALP, CSV, CustomMesh). Ported from Elements/Helium (Serialization.hpp) into
// the orange::io namespace; self-contained (std + Eigen only, header-only).
namespace orange::io {

#define FLT_VALID(x) ((x) < 3.402823466e+36F)

inline int safe_stoi(const std::string& input)
{
	if (input.empty()) return INT_MAX;
	return std::stoi(input);
}

inline float safe_stof(const std::string& input)
{
	if (input.empty()) return FLT_MAX;
	return std::stof(input);
}

inline std::vector<std::string> split(const std::string& input, const std::string& delimiters, bool includeEmptyString = false)
{
	std::vector<std::string> result;
	std::string piece;
	for (auto c : input)
	{
		bool contains = false;
		for (auto d : delimiters)
		{
			if (d == c) { contains = true; break; }
		}

		if (!contains)
		{
			piece += c;
		}
		else
		{
			if (includeEmptyString || !piece.empty())
			{
				result.push_back(piece);
				piece.clear();
			}
		}
	}
	if (!piece.empty()) result.push_back(piece);
	return result;
}

inline void ParseOneLine(
	const std::string& line,
	std::vector<Eigen::Vector3f>& vertices,
	std::vector<Eigen::Vector2f>& uvs,
	std::vector<Eigen::Vector3f>& vertex_normals,
	std::vector<Eigen::Vector4f>& vertex_colors,
	std::vector<Eigen::Vector3i>& faces,
	float scaleX, float scaleY, float scaleZ)
{
	if (line.empty()) return;

	auto words = split(line, " \t");
	if (words.empty()) return;

	if (words[0] == "v")
	{
		float x = safe_stof(words[1]) * scaleX;
		float y = safe_stof(words[2]) * scaleY;
		float z = safe_stof(words[3]) * scaleZ;
		vertices.emplace_back(x, y, z);

		if (words.size() > 4)
		{
			float r = safe_stof(words[4]);
			float g = safe_stof(words[5]);
			float b = safe_stof(words[6]);
			// Assume alpha 1.0 if not present
			vertex_colors.emplace_back(r, g, b, 1.0f);
		}
	}
	else if (words[0] == "vt")
	{
		float u = safe_stof(words[1]);
		float v = safe_stof(words[2]);
		uvs.emplace_back(u, v);
	}
	else if (words[0] == "vn")
	{
		float x = safe_stof(words[1]);
		float y = safe_stof(words[2]);
		float z = safe_stof(words[3]);
		vertex_normals.emplace_back(x, y, z);
	}
	else if (words[0] == "f")
	{
		if (words.size() == 4)
		{
			auto fe0 = split(words[1], "/", true);
			auto fe1 = split(words[2], "/", true);
			auto fe2 = split(words[3], "/", true);

			faces.emplace_back(safe_stoi(fe0[0]), safe_stoi(fe1[0]), safe_stoi(fe2[0]));
		}
	}
}

class HSerializable
{
public:
	virtual ~HSerializable() = default;
	virtual bool Serialize(const std::string& filename) = 0;
	virtual bool Serialize(const std::wstring& filename) { return false; }
	virtual bool Deserialize(const std::string& filename) = 0;
	virtual bool Deserialize(const std::wstring& filename) { return false; }

	virtual inline void AddPoint(float x, float y, float z)
	{
		Eigen::Vector3f p(x, y, z);
		points.push_back(p);
		if (FLT_VALID(x) && FLT_VALID(y) && FLT_VALID(z))
		{
			aabb.extend(p);
		}
	}

	virtual inline void AddPoint(const Eigen::Vector3f& p)
	{
		points.push_back(p);
		if (FLT_VALID(p.x()) && FLT_VALID(p.y()) && FLT_VALID(p.z()))
		{
			aabb.extend(p);
		}
	}

	virtual inline void SwapAxisYZ() {}

	inline const std::vector<Eigen::Vector3f>& GetPoints() const { return points; }
	inline std::vector<Eigen::Vector3f>& GetPoints() { return points; }

	inline Eigen::Vector3f GetAABBMin() const { return aabb.min(); }
	inline Eigen::Vector3f GetAABBMax() const { return aabb.max(); }
	inline Eigen::Vector3f GetAABBCenter() const { return aabb.center(); }

protected:
	std::vector<Eigen::Vector3f> points;
	Eigen::AlignedBox3f aabb;
};

class XYZFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err)
		{
			printf("[Serialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		fprintf(fp, "%llu\n", points.size());
		for (const auto& p : points)
		{
			fprintf(fp, "%.6f %.6f %.6f\n", p.x(), p.y(), p.z());
		}

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		int size = 0;
		fscanf_s(fp, "%d\n", &size);

		points.reserve(size);
		for (int i = 0; i < size; i++)
		{
			float x, y, z;
			fscanf_s(fp, "%f %f %f\n", &x, &y, &z);
			AddPoint(x, y, z);
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points)
		{
			std::swap(p.y(), p.z());
		}
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}
};

class OFFFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err)
		{
			printf("[Serialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		fprintf(fp, "OFF\n");

		auto pointCount = points.size();
		auto faceCount = indices.size();

		fprintf(fp, "%llu %llu %llu\n", pointCount, faceCount, (size_t)0);

		for (size_t i = 0; i < pointCount; i++)
		{
			const auto& p = points[i];
			fprintf(fp, "%4.6f %4.6f %4.6f\n", p.x(), p.y(), p.z());

			if (i % 10000 == 0)
			{
				auto percent = ((double)i / (double)pointCount) * 100.0;
				printf("[%llu / %llu] %f percent\n", i, pointCount, percent);
			}
		}

		for (size_t i = 0; i < faceCount; i++)
		{
			const auto& tri = indices[i];

			if (colors.empty())
			{
				fprintf(fp, "3 %7d %7d %7d 255 255 255\n", tri.x(), tri.y(), tri.z());
			}
			else
			{
				if (tri.x() < (int)colors.size())
				{
					const auto& c = colors[tri.x()];
					auto red = (unsigned char)(c.x() * 255);
					auto green = (unsigned char)(c.y() * 255);
					auto blue = (unsigned char)(c.z() * 255);
					fprintf(fp, "3 %7d %7d %7d %3d %3d %3d\n", tri.x(), tri.y(), tri.z(), red, green, blue);
				}
				else
				{
					fprintf(fp, "3 %7d %7d %7d 255 255 255\n", tri.x(), tri.y(), tri.z());
				}
			}
		}

		if (faceCount == 0 && !colors.empty())
		{
			for (const auto& c : colors)
			{
				auto red = (unsigned char)(c.x() * 255);
				auto green = (unsigned char)(c.y() * 255);
				auto blue = (unsigned char)(c.z() * 255);
				fprintf(fp, "1 %3d %3d %3d\n", red, green, blue);
			}
		}

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		char buffer[1024];
		auto line = fgets(buffer, 1024, fp);
		if (0 != strcmp(line, "OFF\n")) return false;

		line = fgets(buffer, 1024, fp);
		while (line && '#' == line[0])
		{
			line = fgets(buffer, 1024, fp);
		}

		size_t vertexCount = 0;
		size_t triangleCount = 0;
		size_t edgeCount = 0;
		sscanf_s(line, "%llu %llu %llu", &vertexCount, &triangleCount, &edgeCount);

		printf("vertexCount : %llu, triangleCount : %llu\n", vertexCount, triangleCount);

		points.reserve(vertexCount);
		for (size_t i = 0; i < vertexCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if (line)
			{
				if ('#' == line[0]) { i--; continue; }
				float x, y, z;
				sscanf_s(line, "%f %f %f\n", &x, &y, &z);
				AddPoint(x, y, z);
			}
		}

		indices.reserve(triangleCount);
		for (size_t i = 0; i < triangleCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if (line && '#' == line[0]) { i--; continue; }

			unsigned int count, i0, i1, i2;
			sscanf_s(line, "%u %u %u %u\n", &count, &i0, &i1, &i2);
			if (count == 3)
			{
				AddTriangle(i0, i1, i2);
			}
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline const std::vector<Eigen::Vector3i>& GetIndices() const { return indices; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	virtual inline void AddTriangle(unsigned int i0, unsigned int i1, unsigned int i2)
	{
		indices.emplace_back(i0, i1, i2);
	}

	virtual inline void AddColor(float r, float g, float b, float a = 1.0f)
	{
		colors.emplace_back(r, g, b, a);
	}

protected:
	std::vector<Eigen::Vector3i> indices;
	std::vector<Eigen::Vector4f> colors;
};

class STLFormat : public HSerializable
{
public:
	enum class STLDataType
	{
		ASCII,
		BINARY
	};

public:
	virtual bool Serialize(const std::string& filename) override
	{
		if (dataType == STLDataType::ASCII)
		{
			return SerializeASCII(filename);
		}
		else
		{
			return SerializeBinary(filename);
		}
	}

	virtual bool Serialize(const std::wstring& filename) override
	{
		// wstring       .
		//    _wfopen_s    .
		return false;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err) return false;

		char header[6];
		fread(header, 1, 5, fp);
		header[5] = '\0';
		fclose(fp);

		// "solid"  ASCII .
		if (std::string(header) == "solid")
		{
			dataType = STLDataType::ASCII;
			return DeserializeASCII(filename);
		}
		else
		{
			dataType = STLDataType::BINARY;
			return DeserializeBinary(filename);
		}
	}

	virtual bool Deserialize(const std::wstring& filename) override
	{
		return false;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		for (auto& n : faceNormals) std::swap(n.y(), n.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline void SetDataType(STLDataType type) { dataType = type; }
	inline void AddTriangle(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector3f& normal = Eigen::Vector3f::Zero())
	{
		AddPoint(v0);
		AddPoint(v1);
		AddPoint(v2);
		faceNormals.push_back(normal);
	}

protected:
	bool SerializeBinary(const std::string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err) return false;

		// 80 bytes header
		char header[80] = { 0 };
		const char* msg = "Generated by cuTSDF STL Writer";
		memcpy(header, msg, strlen(msg));
		fwrite(header, 1, 80, fp);

		unsigned int numTriangles = (unsigned int)(points.size() / 3);
		fwrite(&numTriangles, 4, 1, fp);

		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			Eigen::Vector3f n = (i < faceNormals.size()) ? faceNormals[i] : Eigen::Vector3f::Zero();
			fwrite(&n.x(), 4, 1, fp);
			fwrite(&n.y(), 4, 1, fp);
			fwrite(&n.z(), 4, 1, fp);

			for (int j = 0; j < 3; ++j)
			{
				const auto& v = points[i * 3 + j];
				fwrite(&v.x(), 4, 1, fp);
				fwrite(&v.y(), 4, 1, fp);
				fwrite(&v.z(), 4, 1, fp);
			}

			unsigned short attributeByteCount = 0;
			fwrite(&attributeByteCount, 2, 1, fp);
		}

		fclose(fp);
		return true;
	}

	bool SerializeASCII(const std::string& filename)
	{
		std::ofstream ofs(filename);
		if (!ofs.is_open()) return false;

		ofs << "solid ASCII_STL" << std::endl;
		size_t numTriangles = points.size() / 3;

		for (size_t i = 0; i < numTriangles; ++i)
		{
			Eigen::Vector3f n = (i < faceNormals.size()) ? faceNormals[i] : Eigen::Vector3f::Zero();
			ofs << "  facet normal " << n.x() << " " << n.y() << " " << n.z() << std::endl;
			ofs << "    outer loop" << std::endl;
			for (int j = 0; j < 3; ++j)
			{
				const auto& v = points[i * 3 + j];
				ofs << "      vertex " << v.x() << " " << v.y() << " " << v.z() << std::endl;
			}
			ofs << "    endloop" << std::endl;
			ofs << "  endfacet" << std::endl;
		}
		ofs << "endsolid ASCII_STL" << std::endl;

		ofs.close();
		return true;
	}

	bool DeserializeBinary(const std::string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err) return false;

		fseek(fp, 80, SEEK_SET);
		unsigned int numTriangles = 0;
		fread(&numTriangles, 4, 1, fp);

		points.reserve(numTriangles * 3);
		faceNormals.reserve(numTriangles);

		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			float nx, ny, nz;
			fread(&nx, 4, 1, fp); fread(&ny, 4, 1, fp); fread(&nz, 4, 1, fp);
			faceNormals.emplace_back(nx, ny, nz);

			for (int j = 0; j < 3; ++j)
			{
				float vx, vy, vz;
				fread(&vx, 4, 1, fp); fread(&vy, 4, 1, fp); fread(&vz, 4, 1, fp);
				AddPoint(vx, vy, vz);
			}

			unsigned short dummy;
			fread(&dummy, 2, 1, fp);
		}

		fclose(fp);
		return true;
	}

	bool DeserializeASCII(const std::string& filename)
	{
		std::ifstream ifs(filename);
		if (!ifs.is_open()) return false;

		std::string line;
		while (std::getline(ifs, line))
		{
			auto words = split(line, " \t");
			if (words.empty()) continue;

			if (words[0] == "facet" && words.size() >= 5)
			{
				faceNormals.emplace_back(safe_stof(words[2]), safe_stof(words[3]), safe_stof(words[4]));
			}
			else if (words[0] == "vertex" && words.size() >= 4)
			{
				AddPoint(safe_stof(words[1]), safe_stof(words[2]), safe_stof(words[3]));
			}
		}

		ifs.close();
		return true;
	}

protected:
	STLDataType dataType = STLDataType::BINARY;
	std::vector<Eigen::Vector3f> faceNormals;
};

class OBJFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		std::ofstream ofs(filename);
		std::stringstream ss;
		ss.precision(6);

		ss << "# cuTSDF::ResourceIO::OBJ" << std::endl;
		for (size_t i = 0; i < points.size(); i++)
		{
			const auto& p = points[i];
			if (colors.size() == points.size())
			{
				const auto& c = colors[i];
				ss << "v " << p.x() << " " << p.y() << " " << p.z() << " " << c.x() << " " << c.y() << " " << c.z() << std::endl;
			}
			else
			{
				ss << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
			}

			if (normals.size() == points.size())
			{
				const auto& n = normals[i];
				ss << "vn " << n.x() << " " << n.y() << " " << n.z() << std::endl;
			}
		}

		for (const auto& uv : uvs)
		{
			ss << "vt " << uv.x() << " " << uv.y() << std::endl;
		}

		bool has_uv = !uvs.empty();
		bool has_vn = !normals.empty();

		for (size_t i = 0; i < indices.size(); i++)
		{
			const auto& face = indices[i];
			if (has_uv && has_vn)
			{
				ss << "f "
					<< face.x() << "/" << face.x() << "/" << face.x() << " "
					<< face.y() << "/" << face.y() << "/" << face.y() << " "
					<< face.z() << "/" << face.z() << "/" << face.z() << std::endl;
			}
			else if (has_uv)
			{
				ss << "f "
					<< face.x() << "/" << face.x() << " "
					<< face.y() << "/" << face.y() << " "
					<< face.z() << "/" << face.z() << std::endl;
			}
			else if (has_vn)
			{
				ss << "f "
					<< face.x() << "//" << face.x() << " "
					<< face.y() << "//" << face.y() << " "
					<< face.z() << "//" << face.z() << std::endl;
			}
			else
			{
				ss << "f " << face.x() << " " << face.y() << " " << face.z() << std::endl;
			}

			if (i % 10000 == 0)
			{
				auto percent = ((double)i / (double)indices.size()) * 100.0;
				printf("[%llu / %llu] %f percent\n", i, indices.size(), percent);
			}
		}

		ofs << ss.rdbuf();
		ofs.close();
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		std::ifstream ifs(filename);
		if (!ifs.is_open())
		{
			printf("filename : %s is not open\n", filename.c_str());
			return false;
		}

		std::stringstream buffer;
		buffer << ifs.rdbuf();

		std::string line;
		while (buffer.good())
		{
			getline(buffer, line);
			ParseOneLine(line, points, uvs, normals, colors, indices, 1.0f, 1.0f, 1.0f);
			if (!points.empty()) aabb.extend(points.back());
		}
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		for (auto& n : normals) std::swap(n.y(), n.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline const std::vector<Eigen::Vector3f>& GetNormals() const { return normals; }
	inline const std::vector<Eigen::Vector3i>& GetIndices() const { return indices; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	virtual inline void AddUV(float u, float v) { uvs.emplace_back(u, v); }
	virtual inline void AddNormal(float x, float y, float z) { normals.emplace_back(x, y, z); }
	virtual inline void AddTriangle(int i0, int i1, int i2) { indices.emplace_back(i0, i1, i2); }
	virtual inline void AddColor(float r, float g, float b, float a = 1.0f) { colors.emplace_back(r, g, b, a); }

protected:
	std::vector<Eigen::Vector2f> uvs;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector3i> indices;
	std::vector<Eigen::Vector4f> colors;
};

class PLYFormat : public HSerializable
{
public:
	enum class PLYDataType
	{
		ASCII,
		BINARY
	};

public:
	virtual bool Serialize(const std::string& filename) override
	{
		std::ofstream ofs(filename, std::ios::out | std::ios::binary);
		return Serialize(ofs);
	}

	virtual bool Serialize(const std::wstring& filename) override
	{
		std::ofstream ofs(filename, std::ios::out | std::ios::binary);
		return Serialize(ofs);
	}

	bool Serialize(std::ofstream& ofs)
	{
		if (!ofs.is_open()) return false;

		std::stringstream ssHeader;
		ssHeader << "ply" << std::endl;

		if (dataType == PLYDataType::ASCII)
			ssHeader << "format ascii 1.0" << std::endl;
		else
			ssHeader << "format binary_little_endian 1.0" << std::endl;

		ssHeader << "element vertex " << points.size() << std::endl;
		ssHeader << "property float x" << std::endl;
		ssHeader << "property float y" << std::endl;
		ssHeader << "property float z" << std::endl;

		if (normals.size() == points.size())
		{
			ssHeader << "property float nx" << std::endl;
			ssHeader << "property float ny" << std::endl;
			ssHeader << "property float nz" << std::endl;
		}
		if (colors.size() == points.size())
		{
			ssHeader << "property uchar red" << std::endl;
			ssHeader << "property uchar green" << std::endl;
			ssHeader << "property uchar blue" << std::endl;
			if (useAlpha) ssHeader << "property uchar alpha" << std::endl;
		}
		if (uvs.size() == points.size())
		{
			ssHeader << "property float u" << std::endl;
			ssHeader << "property float v" << std::endl;
		}
		if (labels.size() == points.size())
		{
			ssHeader << "property int label" << std::endl;
		}
		if (deepLearningClasses.size() == points.size())
		{
			ssHeader << "property int deepLearningClass" << std::endl;
		}

		if (!lineIndices.empty())
		{
			ssHeader << "element edge " << lineIndices.size() << std::endl;
			ssHeader << "property int vertex1" << std::endl;
			ssHeader << "property int vertex2" << std::endl;
		}

		if (!triangleIndices.empty())
		{
			ssHeader << "element face " << triangleIndices.size() << std::endl;
			ssHeader << "property list uchar int vertex_indices" << std::endl;
		}

		ssHeader << "end_header" << std::endl;
		ofs.write(ssHeader.str().c_str(), ssHeader.str().length());

		if (dataType == PLYDataType::ASCII)
		{
			std::stringstream ss;
			ss.precision(6);
			for (size_t i = 0; i < points.size(); i++)
			{
				const auto& p = points[i];
				ss << p.x() << " " << p.y() << " " << p.z() << " ";

				if (normals.size() == points.size())
				{
					const auto& n = normals[i];
					ss << n.x() << " " << n.y() << " " << n.z() << " ";
				}

				if (colors.size() == points.size())
				{
					const auto& c = colors[i];
					ss << (int)(c.x() * 255) << " " << (int)(c.y() * 255) << " " << (int)(c.z() * 255) << " ";
					if (useAlpha) ss << (int)(c.w() * 255) << " ";
				}

				if (uvs.size() == points.size())
				{
					const auto& uv = uvs[i];
					ss << uv.x() << " " << uv.y() << " ";
				}

				if (labels.size() == points.size()) ss << labels[i] << " ";
				if (deepLearningClasses.size() == points.size()) ss << deepLearningClasses[i] << " ";

				ss << std::endl;

				if (i % 5000 == 0) { ofs << ss.rdbuf(); ss.str(std::string()); ss.clear(); }
			}
			ofs << ss.rdbuf();

			for (const auto& line : lineIndices)
				ofs << line.x() << " " << line.y() << std::endl;

			for (const auto& tri : triangleIndices)
				ofs << "3 " << tri.x() << " " << tri.y() << " " << tri.z() << std::endl;
		}
		else // BINARY
		{
			for (size_t i = 0; i < points.size(); i++)
			{
				const auto& p = points[i];
				ofs.write(reinterpret_cast<const char*>(&p.x()), sizeof(float));
				ofs.write(reinterpret_cast<const char*>(&p.y()), sizeof(float));
				ofs.write(reinterpret_cast<const char*>(&p.z()), sizeof(float));

				if (normals.size() == points.size())
				{
					const auto& n = normals[i];
					ofs.write(reinterpret_cast<const char*>(&n.x()), sizeof(float));
					ofs.write(reinterpret_cast<const char*>(&n.y()), sizeof(float));
					ofs.write(reinterpret_cast<const char*>(&n.z()), sizeof(float));
				}

				if (colors.size() == points.size())
				{
					const auto& c = colors[i];
					unsigned char r = (unsigned char)(c.x() * 255);
					unsigned char g = (unsigned char)(c.y() * 255);
					unsigned char b = (unsigned char)(c.z() * 255);
					ofs.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
					ofs.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
					ofs.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
					if (useAlpha) {
						unsigned char a = (unsigned char)(c.w() * 255);
						ofs.write(reinterpret_cast<const char*>(&a), sizeof(unsigned char));
					}
				}

				if (uvs.size() == points.size())
				{
					const auto& uv = uvs[i];
					ofs.write(reinterpret_cast<const char*>(&uv.x()), sizeof(float));
					ofs.write(reinterpret_cast<const char*>(&uv.y()), sizeof(float));
				}

				if (labels.size() == points.size())
				{
					ofs.write(reinterpret_cast<const char*>(&labels[i]), sizeof(int));
				}
				if (deepLearningClasses.size() == points.size())
				{
					ofs.write(reinterpret_cast<const char*>(&deepLearningClasses[i]), sizeof(int));
				}
			}

			// Edges (assuming raw int pairs if not list, standard PLY usually uses vertex indices)
			for (const auto& line : lineIndices)
			{
				// Note: Property definition above was "property int vertex1", "property int vertex2" (scalars)
				// If we want list property, we should change header. 
				// Consistently writing scalars as per header definition in Serialize().
				int v1 = line.x();
				int v2 = line.y();
				ofs.write(reinterpret_cast<const char*>(&v1), sizeof(int));
				ofs.write(reinterpret_cast<const char*>(&v2), sizeof(int));
			}

			// Faces
			unsigned char listCount = 3;
			for (const auto& tri : triangleIndices)
			{
				ofs.write(reinterpret_cast<const char*>(&listCount), sizeof(unsigned char));
				int i0 = tri.x();
				int i1 = tri.y();
				int i2 = tri.z();
				ofs.write(reinterpret_cast<const char*>(&i0), sizeof(int));
				ofs.write(reinterpret_cast<const char*>(&i1), sizeof(int));
				ofs.write(reinterpret_cast<const char*>(&i2), sizeof(int));
			}
		}

		ofs.close();
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		std::ifstream ifs(filename, std::ios::in | std::ios::binary);
		return Deserialize(ifs);
	}

	virtual bool Deserialize(const std::wstring& filename) override
	{
		std::ifstream ifs(filename, std::ios::in | std::ios::binary);
		return Deserialize(ifs);
	}

	virtual bool Deserialize(std::ifstream& ifs)
	{
		if (!ifs.is_open()) return false;

		std::string line;
		std::vector<std::string> elementNames;
		std::vector<size_t> elementCounts;
		std::vector<bool> listTypeInfo;
		// Store property types: name -> type string
		struct PropInfo { std::string name; std::string type; };
		std::vector<std::vector<PropInfo>> elementProperties;
		bool isBinary = false;

		// 1. Header Parsing
		while (true)
		{
			// Handle mixed line endings by reading char by char or getline and stripping \r
			std::getline(ifs, line);
			if (!line.empty() && line.back() == '\r') line.pop_back();

			auto words = split(line, " \t");
			if (words.empty()) continue;

			if (words[0] == "format")
			{
				if (words[1] == "binary_little_endian") isBinary = true;
				else isBinary = false;
			}
			else if (words[0] == "element")
			{
				elementNames.push_back(words[1]);
				elementCounts.push_back(atoi(words[2].c_str()));
				elementProperties.emplace_back();
				listTypeInfo.push_back(false);
			}
			else if (words[0] == "property")
			{
				size_t index = elementNames.size() - 1;
				if (words[1] == "list")
				{
					listTypeInfo[index] = true;
					// Format: property list <count_type> <index_type> <name>
					elementProperties[index].push_back({ words[4], "list" });
				}
				else
				{
					// Format: property <type> <name>
					std::string name = words[2];
					if (name == "alpha" || name == "a") useAlpha = true;
					elementProperties[index].push_back({ name, words[1] });
				}
			}
			else if (words[0] == "end_header") break;
		}

		SetDataType(isBinary ? PLYDataType::BINARY : PLYDataType::ASCII);

		// Helper lambda to read binary values
		auto ReadBinVal = [&](const std::string& type) -> float {
			if (type == "float" || type == "float32") { float v; ifs.read((char*)&v, 4); return v; }
			else if (type == "int" || type == "int32") { int v; ifs.read((char*)&v, 4); return (float)v; }
			else if (type == "uchar" || type == "uint8") { unsigned char v; ifs.read((char*)&v, 1); return (float)v; }
			// Simplified: assuming standard PLY types. Add double/short/etc if needed.
			return 0.0f;
			};

		// 2. Body Parsing
		for (size_t i = 0; i < elementNames.size(); i++)
		{
			bool isList = listTypeInfo[i];
			size_t count = elementCounts[i];

			if (elementNames[i] == "vertex")
			{
				points.reserve(count);
				for (size_t j = 0; j < count; j++)
				{
					float x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0, u = 0, v = 0;
					float r = 0, g = 0, b = 0, a = 1.0f;
					int label = 0, dlClass = 0;
					bool hasNormal = false, hasColor = false, hasUV = false, hasLabel = false, hasDL = false;

					if (isBinary)
					{
						for (const auto& prop : elementProperties[i])
						{
							float val = ReadBinVal(prop.type);
							if (prop.name == "x") x = val;
							else if (prop.name == "y") y = val;
							else if (prop.name == "z") z = val;
							else if (prop.name == "nx") { nx = val; hasNormal = true; }
							else if (prop.name == "ny") { ny = val; hasNormal = true; }
							else if (prop.name == "nz") { nz = val; hasNormal = true; }
							else if (prop.name == "red") { r = val / 255.0f; hasColor = true; }
							else if (prop.name == "green") { g = val / 255.0f; hasColor = true; }
							else if (prop.name == "blue") { b = val / 255.0f; hasColor = true; }
							else if (prop.name == "alpha") { a = val / 255.0f; hasColor = true; }
							else if (prop.name == "u") { u = val; hasUV = true; }
							else if (prop.name == "v") { v = val; hasUV = true; }
							else if (prop.name == "label") { label = (int)val; hasLabel = true; }
							else if (prop.name == "deepLearningClass") { dlClass = (int)val; hasDL = true; }
						}
					}
					else // ASCII
					{
						std::getline(ifs, line);
						auto words = split(line, " \t");
						for (size_t k = 0; k < words.size() && k < elementProperties[i].size(); k++)
						{
							const auto& prop = elementProperties[i][k];
							float val = (float)atof(words[k].c_str());
							if (prop.name == "x") x = val;
							else if (prop.name == "y") y = val;
							else if (prop.name == "z") z = val;
							else if (prop.name == "nx") { nx = val; hasNormal = true; }
							else if (prop.name == "ny") { ny = val; hasNormal = true; }
							else if (prop.name == "nz") { nz = val; hasNormal = true; }
							else if (prop.name == "red") { r = val / 255.0f; hasColor = true; }
							else if (prop.name == "green") { g = val / 255.0f; hasColor = true; }
							else if (prop.name == "blue") { b = val / 255.0f; hasColor = true; }
							else if (prop.name == "alpha") { a = val / 255.0f; hasColor = true; }
							else if (prop.name == "u") { u = val; hasUV = true; }
							else if (prop.name == "v") { v = val; hasUV = true; }
							else if (prop.name == "label") { label = (int)val; hasLabel = true; }
							else if (prop.name == "deepLearningClass") { dlClass = (int)val; hasDL = true; }
						}
					}

					AddPoint(x, y, z);
					if (hasNormal) AddNormal(nx, ny, nz);
					if (hasColor) AddColor(r, g, b, a);
					if (hasUV) AddUV(u, v);
					if (hasLabel) labels.push_back(label);
					if (hasDL) deepLearningClasses.push_back(dlClass);
				}
			}
			else if (elementNames[i] == "face" || elementNames[i] == "edge")
			{
				for (size_t j = 0; j < count; j++)
				{
					if (isList) // Standard face definition (list uchar int)
					{
						int listSize = 0;
						if (isBinary) {
							unsigned char c; ifs.read((char*)&c, 1); listSize = c;
						}
						else {
							std::string token; ifs >> token; listSize = atoi(token.c_str());
						}

						std::vector<int> idxs(listSize);
						for (int k = 0; k < listSize; ++k) {
							if (isBinary) { int v; ifs.read((char*)&v, 4); idxs[k] = v; }
							else { std::string token; ifs >> token; idxs[k] = atoi(token.c_str()); }
						}

						if (elementNames[i] == "face" && listSize == 3) AddTriangle(idxs[0], idxs[1], idxs[2]);
						// Edge handling for lists could be added here
					}
					else // Scalar properties (e.g. edge with vertex1, vertex2)
					{
						int v1 = 0, v2 = 0;
						if (isBinary) {
							for (const auto& prop : elementProperties[i]) {
								float val = ReadBinVal(prop.type); // assuming int stored
								if (prop.name == "vertex1") v1 = (int)val;
								else if (prop.name == "vertex2") v2 = (int)val;
							}
						}
						else {
							std::getline(ifs, line);
							auto words = split(line, " \t");
							for (size_t k = 0; k < words.size(); ++k) {
								if (elementProperties[i][k].name == "vertex1") v1 = atoi(words[k].c_str());
								else if (elementProperties[i][k].name == "vertex2") v2 = atoi(words[k].c_str());
							}
						}
						if (elementNames[i] == "edge") AddLineIndex(v1, v2);
					}
				}
				// If reading ASCII via >> tokens, consume rest of line if mixed
				if (!isBinary && isList) { std::string dummy; std::getline(ifs, dummy); }
			}
			else // Unknown elements
			{
				// Skip unknown logic is complex for binary (need size calculation), 
				// for ASCII simple skip line. 
				if (!isBinary) {
					std::string dummy;
					for (size_t j = 0; j < count; ++j) std::getline(ifs, dummy);
				}
				// Binary skip omitted for brevity, assuming standard files
			}
		}
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points)
		{
			std::swap(p.y(), p.z());
			p.z() = -p.z();
		}
		for (auto& n : normals) std::swap(n.y(), n.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline std::vector<Eigen::Vector3f>& GetNormals() { return normals; }
	inline const std::vector<Eigen::Vector3f>& GetNormals() const { return normals; }
	inline const std::vector<Eigen::Vector2i>& GetLineIndices() const { return lineIndices; }
	inline const std::vector<Eigen::Vector3i>& GetTriangleIndices() const { return triangleIndices; }
	inline std::vector<Eigen::Vector4f>& GetColors() { return colors; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	inline PLYDataType GetDataType() const { return dataType; }
	inline void SetDataType(PLYDataType type) { dataType = type; }

	virtual inline void AddUV(float u, float v) { uvs.emplace_back(u, v); }
	virtual inline void AddNormal(float x, float y, float z) { normals.emplace_back(x, y, z); }
	virtual inline void AddLineIndex(int i0, int i1) { lineIndices.emplace_back(i0, i1); }
	virtual inline void AddTriangle(int i0, int i1, int i2) { triangleIndices.emplace_back(i0, i1, i2); }
	virtual inline void AddColor(float r, float g, float b, float a = 1.0f) { colors.emplace_back(r, g, b, a); }

	virtual void AddCube(float cx, float cy, float cz, float nx, float ny, float nz, float r, float g, float b, float a, float scale)
	{
		float h = scale * 0.5f;
		Eigen::Vector3f center(cx, cy, cz);

		Eigen::Vector3f verts[8] = {
			{cx - h, cy - h, cz - h}, {cx + h, cy - h, cz - h},
			{cx + h, cy + h, cz - h}, {cx - h, cy + h, cz - h},
			{cx - h, cy - h, cz + h}, {cx + h, cy - h, cz + h},
			{cx + h, cy + h, cz + h}, {cx - h, cy + h, cz + h}
		};

		unsigned int base = (unsigned int)points.size();

		for (int i = 0; i < 8; ++i)
		{
			AddPoint(verts[i]);
			AddNormal(nx, ny, nz);
			AddColor(r, g, b, a);
		}

		static const int cube_tris[12][3] = {
			{0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6},
			{0, 4, 5}, {0, 5, 1}, {2, 6, 7}, {2, 7, 3},
			{0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2}
		};

		for (int i = 0; i < 12; ++i)
		{
			AddTriangle(base + cube_tris[i][0], base + cube_tris[i][1], base + cube_tris[i][2]);
		}
	}

	virtual void AddData(float* pos, float* nrm, float* col, unsigned int count, bool alpha)
	{
		useAlpha = alpha;
		for (unsigned int i = 0; i < count; ++i)
		{
			AddPoint(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
			AddNormal(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]);
			if (alpha) AddColor(col[i * 4], col[i * 4 + 1], col[i * 4 + 2], col[i * 4 + 3]);
			else AddColor(col[i * 3], col[i * 3 + 1], col[i * 3 + 2], 1.0f);
		}
	}

protected:
	PLYDataType dataType = PLYDataType::BINARY;

	std::vector<Eigen::Vector2f> uvs;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector2i> lineIndices;
	std::vector<Eigen::Vector3i> triangleIndices;
	std::vector<Eigen::Vector4f> colors;
	std::vector<int> labels;
	std::vector<int> deepLearningClasses;
	bool useAlpha = false;
};

class PTSFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		std::ofstream ofs(filename);
		if (!ofs.is_open())
		{
			return false;
		}

		ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
		ofs << "<Teeth>" << std::endl;
		ofs << " <ScanTooth>" << std::endl;
		ofs << "  <Number>0</Number>" << std::endl; //  0 
		ofs << "  <PreparationMargin>" << std::endl;

		for (const auto& p : points)
		{
			ofs << "   <Vec>" << std::endl;
			ofs << "    <x>" << p.x() << "</x>" << std::endl;
			ofs << "    <y>" << p.y() << "</y>" << std::endl;
			ofs << "    <z>" << p.z() << "</z>" << std::endl;
			ofs << "   </Vec>" << std::endl;
		}

		ofs << "  </PreparationMargin>" << std::endl;
		ofs << "  <Axis>" << std::endl;
		ofs << "   <x>" << axis.x() << "</x>" << std::endl;
		ofs << "   <y>" << axis.y() << "</y>" << std::endl;
		ofs << "   <z>" << axis.z() << "</z>" << std::endl;
		ofs << "  </Axis>" << std::endl;
		ofs << " </ScanTooth>" << std::endl;
		ofs << "</Teeth>" << std::endl;

		ofs.close();
		return true;
	}

	virtual bool Serialize(const std::wstring& filename) override
	{
		return false;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		std::ifstream ifs(filename);
		if (!ifs.is_open())
		{
			return false;
		}

		std::string line;
		bool inMargin = false;
		bool inAxis = false;
		float x = 0, y = 0, z = 0;

		while (std::getline(ifs, line))
		{
			if (line.find("<PreparationMargin>") != std::string::npos)
			{
				inMargin = true;
				continue;
			}
			if (line.find("</PreparationMargin>") != std::string::npos)
			{
				inMargin = false;
				continue;
			}
			if (line.find("<Axis>") != std::string::npos)
			{
				inAxis = true;
				continue;
			}
			if (line.find("</Axis>") != std::string::npos)
			{
				inAxis = false;
				continue;
			}

			if (line.find("<x>") != std::string::npos)
			{
				x = ExtractValue(line);
			}
			else if (line.find("<y>") != std::string::npos)
			{
				y = ExtractValue(line);
			}
			else if (line.find("<z>") != std::string::npos)
			{
				z = ExtractValue(line);
				if (inMargin)
				{
					AddPoint(x, y, z);
				}
				else if (inAxis)
				{
					axis = Eigen::Vector3f(x, y, z);
				}
			}
		}

		ifs.close();
		return true;
	}

	virtual bool Deserialize(const std::wstring& filename) override
	{
		return false;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points)
		{
			std::swap(p.y(), p.z());
		}
		std::swap(axis.y(), axis.z());

		aabb.setEmpty();
		for (const auto& p : points)
		{
			aabb.extend(p);
		}
	}

	inline void SetAxis(const Eigen::Vector3f& a)
	{
		axis = a;
	}

	inline Eigen::Vector3f GetAxis() const
	{
		return axis;
	}

private:
	float ExtractValue(const std::string& line)
	{
		size_t start = line.find('>') + 1;
		size_t end = line.find('<', start);
		if (start != std::string::npos && end != std::string::npos)
		{
			return std::stof(line.substr(start, end - start));
		}
		return 0.0f;
	}

protected:
	Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
};

struct EigenPoint
{
	Eigen::Vector3f position;
	Eigen::Vector3f normal;
	Eigen::Vector3f color; // or Vector4f
};

template<typename Point = EigenPoint>
class ALPFormat
{
public:
	bool Serialize(const std::string& filename)
	{
		std::ofstream ofs(filename, std::ios::out | std::ios::binary);
		if (!ofs.is_open()) return false;

		unsigned long nop = (unsigned long)points.size();
		unsigned int pointSize = sizeof(Point);

		ofs.write((char*)&nop, sizeof(unsigned long));
		ofs.write((char*)&pointSize, sizeof(unsigned int));

		if (nop > 0)
			ofs.write((char*)points.data(), nop * pointSize);

		ofs.close();
		return true;
	}

	bool Deserialize(const std::string& filename)
	{
		std::ifstream ifs(filename, std::ios::in | std::ios::binary);
		if (!ifs.is_open()) return false;

		unsigned long nop = 0;
		unsigned int pointSize = 0;

		ifs.read((char*)&nop, sizeof(unsigned long));
		ifs.read((char*)&pointSize, sizeof(unsigned int));

		if (pointSize != sizeof(Point))
		{
			// Size mismatch handling
			ifs.close();
			return false;
		}

		points.resize(nop);
		ifs.read((char*)points.data(), nop * pointSize);

		// Update AABB
		aabb.setEmpty();
		// Assumes Point has 'position' member which is Eigen::Vector3f
		for (const auto& p : points) aabb.extend(p.position);

		ifs.close();
		return true;
	}

	void AddPoint(const Point& point)
	{
		std::lock_guard<std::mutex> lock(points_mutex);
		points.push_back(point);
		aabb.extend(point.position);
	}

	void AddPoints(const std::vector<Point>& inputPoints)
	{
		std::lock_guard<std::mutex> lock(points_mutex);
		points.insert(points.end(), inputPoints.begin(), inputPoints.end());
		for (const auto& p : inputPoints) aabb.extend(p.position);
	}

	const std::vector<Point>& GetPoints() const
	{
		std::lock_guard<std::mutex> lock(points_mutex);
		return points;
	}

	// Helper to convert from PLY to internal points
	void FromPLY(const PLYFormat& ply)
	{
		const auto& plyPoints = ply.GetPoints();
		const auto& plyNormals = ply.GetNormals();
		const auto& plyColors = ply.GetColors();

		size_t count = plyPoints.size();
		points.reserve(points.size() + count);

		for (size_t i = 0; i < count; i++)
		{
			Point p;
			// Assuming Point struct structure:
			p.position = plyPoints[i];
			if (i < plyNormals.size()) p.normal = plyNormals[i];
			else p.normal = Eigen::Vector3f::Zero();

			if (i < plyColors.size()) p.color = plyColors[i].head<3>(); // Taking RGB
			else p.color = Eigen::Vector3f::Ones();

			AddPoint(p);
		}
	}

	inline Eigen::Vector3f GetAABBMin() { return aabb.min(); }
	inline Eigen::Vector3f GetAABBMax() { return aabb.max(); }
	inline Eigen::Vector3f GetAABBCenter() { return aabb.center(); }

protected:
	mutable std::mutex points_mutex;
	std::vector<Point> points;
	Eigen::AlignedBox3f aabb;
};

class CSVFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err) return false;

		fprintf(fp, "%llu\n", points.size());
		fprintf(fp, "X, Y, Z\n");
		for (const auto& p : points)
		{
			fprintf(fp, "%.6f, %.6f, %.6f\n", p.x(), p.y(), p.z());
		}

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err) return false;

		char buffer[1024];
		// Skip header lines potentially? Original skipped one line
		fgets(buffer, sizeof(buffer), fp); // Size?
		fgets(buffer, sizeof(buffer), fp); // Header X,Y,Z?

		while (fgets(buffer, sizeof(buffer), fp)) {
			float x, y, z;
			if (sscanf_s(buffer, "%f,%f,%f", &x, &y, &z) == 3) {
				AddPoint(x, y, z);
			}
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}
};

class CustomMeshFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err) return false;

		fprintf_s(fp, "%llu\n", points.size());
		for (const auto& p : points)
			fprintf(fp, "%f, %f, %f\n", p.x(), p.y(), p.z());

		fprintf_s(fp, "%llu\n", normals.size());
		for (const auto& n : normals)
			fprintf(fp, "%f, %f, %f\n", n.x(), n.y(), n.z());

		fprintf_s(fp, "%llu\n", colors.size());
		for (const auto& c : colors)
			fprintf(fp, "%f, %f, %f\n", c.x(), c.y(), c.z());

		fprintf_s(fp, "%llu\n", indices.size());
		for (const auto& tri : indices)
			fprintf(fp, "%d, %d, %d\n", tri.x(), tri.y(), tri.z());

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err) return false;

		char buffer[1024];

		size_t nop = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &nop);
		points.reserve(nop);
		for (size_t i = 0; i < nop; ++i) {
			float x, y, z;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%f, %f, %f\n", &x, &y, &z);
			AddPoint(x, y, z);
		}

		size_t non = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &non);
		normals.reserve(non);
		for (size_t i = 0; i < non; ++i) {
			float x, y, z;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%f, %f, %f\n", &x, &y, &z);
			AddNormal(x, y, z);
		}

		size_t noc = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &noc);
		colors.reserve(noc);
		for (size_t i = 0; i < noc; ++i) {
			float r, g, b;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%f, %f, %f\n", &r, &g, &b);
			AddColor(r, g, b);
		}

		size_t noi = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &noi);
		indices.reserve(noi);
		for (size_t i = 0; i < noi; ++i) {
			int i0, i1, i2;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%d, %d, %d\n", &i0, &i1, &i2);
			AddTriangle(i0, i1, i2);
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		for (auto& n : normals) std::swap(n.y(), n.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline const std::vector<Eigen::Vector3i>& GetIndices() const { return indices; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	virtual inline void AddNormal(float x, float y, float z) { normals.emplace_back(x, y, z); }
	virtual inline void AddNormal(const Eigen::Vector3f& n) { normals.push_back(n); }
	virtual inline void AddTriangle(int i0, int i1, int i2) { indices.emplace_back(i0, i1, i2); }
	virtual inline void AddColor(float r, float g, float b, float a = 1.0f) { colors.emplace_back(r, g, b, a); }

protected:
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector3i> indices;
	std::vector<Eigen::Vector4f> colors;
};

#undef FLT_VALID

} // namespace orange::io
