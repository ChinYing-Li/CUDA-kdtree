// OBJCORE- A Obj Mesh Library by Yining Karl Li
// This file is part of OBJCORE, Coyright (c) 2012 Yining Karl Li
// Modified by Chin-Ying Li, 2021

#ifndef OBJ
#define OBJ

#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class obj{
private:
  std::vector<glm::vec4> points;
  std::vector<std::vector<int> > faces;
  std::vector<std::vector<int> > facenormals;
  std::vector<std::vector<int> > facetextures;
  std::vector<float*> faceboxes;   //bounding boxes for each face are stored in vbo-format!
  std::vector<glm::vec4> normals;
  std::vector<glm::vec4> texturecoords;
	int vbosize;
	int nbosize;
	int cbosize;
	int ibosize;
	int tbosize;
  std::vector<float> vbo_vec;
  std::vector<float> nbo_vec;
  std::vector<float> cbo_vec;
  std::vector<int> ibo_vec;
  std::vector<float> tbo_vec;

  /* Avoid using C-array!
	float* vbo;
	float* nbo;
	float* cbo;
	int* ibo;
	float* tbo;
  */

  std::vector<float> boundingbox; // VBO-formated bounding box
	float top;
	glm::vec3 defaultColor;
	float xmax; float xmin; float ymax; float ymin; float zmax; float zmin; 
	bool maxminSet;
public:
	obj();
	~obj();  

	//-------------------------------
	//-------Mesh Operations---------
	//-------------------------------
	void buildVBOs();
	void addPoint(glm::vec3);
  void addFace(std::vector<int>);
	void addNormal(glm::vec3);
	void addTextureCoord(glm::vec3);
  void addFaceNormal(std::vector<int>);
  void addFaceTexture(std::vector<int>);
	void compareMaxMin(float, float, float);
  bool isConvex(std::vector<int>&);
	void recenter();

	//-------------------------------
  //------- Getter / Setter ------
	//-------------------------------
  std::vector<float> getBoundingBox();    //returns vbo-formatted bounding box
	float getTop();
	void setColor(glm::vec3);
	glm::vec3 getColor();

  const std::vector<float>& getVBO() const;
  const std::vector<float>& getCBO() const;
  const std::vector<float>& getNBO() const;
  const std::vector<int>& getIBO() const;
  const std::vector<float>& getTBO() const;

  int getVBOsize() const;
  int getNBOsize() const;
  int getIBOsize() const;
  int getCBOsize() const;
  int getTBOsize() const;

  std::vector<glm::vec4>* getPoints();
  std::vector<std::vector<int> >* getFaces();
  std::vector<std::vector<int> >* getFaceNormals();
  std::vector<std::vector<int> >* getFaceTextures();
  std::vector<glm::vec4>* getNormals();
  std::vector<glm::vec4>* getTextureCoords();
  std::vector<float*>* getFaceBoxes();
};

#endif
