#ifndef FDECS_HPP
#define FDECS_HPP
/*
* Forward Declaration of AST Nodes
* - also used as reference point
*/

#include <vector>
#include <memory>
#include <string>
#include <iostream>

class ASTnode;

class StmtNode;

class ExprNode;

/*
class ProgramNode : public ASTnode;
//class ExternNode;

class FloatLitNode;
class BoolLitNode;
class IntLitNode;
//class TypeNode;

class ExprNode;
class UnaryNode;
class BinaryNode;

class BlockNode;
class StmtNode;
class VarDecNode;
class VarCallNode;

class DeclNode;
class FuncDecNode;
class FuncCallNode;
class LocalDecNode;

class WhileNode;
class IfNode;
class ReturnNode;
class ParamNode;
*/

enum TYPES{
  void_T = 0,
  int_T,
  float_T,
  boolean_T
};



 // Forward Declaration of Functions



 /*
static std::unique_ptr<ProgramNode> program();
static std::vector<std::unique_ptr<FuncDecNode>> extern_list();
static void externp_list(std::vector<std::unique_ptr<FuncDecNode>> &externs);
static std::unique_ptr<FuncDecNode> extern_sing();
static std::vector<std::unique_ptr<DeclNode>> decl_list();
static void declp_list(std::vector<std::unique_ptr<DeclNode>> &decls);
static std::unique_ptr<DeclNode> decl();
static std::unique_ptr<VarDecNode> var_decl();

//TYPES type_spec();
//TYPES var_type();

static std::unique_ptr<FuncDecNode> func_decl();

//static std::unique_ptr<ParamsNode> param_list();
static std::vector<std::unique_ptr<ParamNode>> params();
static std::unique_ptr<ParamNode> param();

static std::unique_ptr<BlockNode> block();
static std::unique_ptr<VarDecNode> local_decl();
void local_decls();

static std::unique_ptr<StmtNode> stmt();

static void stmt_list(std::vector<std::unique_ptr<StmtNode>> &stmt_list);

static std::unique_ptr<WhileNode> while_stmt();
static std::unique_ptr<IfNode> if_stmt();
static std::unique_ptr<ReturnNode> return_stmt();

static std::unique_ptr<ExprNode> logor();
static std::unique_ptr<ExprNode> logorp(std::unique_ptr<ExprNode> lhs);
static std::unique_ptr<ExprNode> logand();
static std::unique_ptr<ExprNode> logandp(std::unique_ptr<ExprNode> lhs);
static std::unique_ptr<ExprNode> logeq();
static std::unique_ptr<ExprNode> logeqp(std::unique_ptr<ExprNode> lhs);
static std::unique_ptr<ExprNode> relexpr();
static std::unique_ptr<ExprNode> relexprp(std::unique_ptr<ExprNode> lhs);
static std::unique_ptr<ExprNode> addmin();
static std::unique_ptr<ExprNode> addminp(std::unique_ptr<ExprNode> lhs);
static std::unique_ptr<ExprNode> multdiv();
static std::unique_ptr<ExprNode> multdivp(std::unique_ptr<ExprNode> lhs);
static std::unique_ptr<ExprNode> neg();
static std::unique_ptr<ExprNode> liter();

*/

#endif