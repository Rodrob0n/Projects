#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string.h>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"


// ----------
 #include "fdecs.hpp"
#include "token.hpp"

// ----------

using namespace llvm;
using namespace llvm::sys;

FILE *pFile;

/*
CODEGEN TYPING = GETtPE THEN COMEPATE

FOR ALLOCA INST TYPE IS SPECIAL UNLESS YOU DO GETALLOCATED TYPE
OR GET GLOBAL TYEP FOR GLOBAL VARIABLES
*/

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns one of these for known things.


static std::string IdentifierStr; // Filled in if IDENT
static int IntVal;                // Filled in if INT_LIT
static bool BoolVal;              // Filled in if BOOL_LIT
static float FloatVal;            // Filled in if FLOAT_LIT
static std::string StringVal;     // Filled in if String Literal
static int lineNo, columnNo;
static bool alwaysReturn;


static TOKEN returnTok(std::string lexVal, int tok_type) {
  TOKEN return_tok;
  return_tok.lexeme = lexVal;
  return_tok.type = tok_type;
  return_tok.lineNo = lineNo;
  return_tok.columnNo = columnNo - lexVal.length() - 1;
  return return_tok;
}

// Read file line by line -- or look for \n and if found add 1 to line number
// and reset column number to 0
/// gettok - Return the next token from standard input.
static TOKEN gettok() {

  static int LastChar = ' ';
  static int NextChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar)) {
    if (LastChar == '\n' || LastChar == '\r') {
      lineNo++;
      columnNo = 1;
    }
    LastChar = getc(pFile);
    columnNo++;
  }

  if (isalpha(LastChar) ||
      (LastChar == '_')) { // identifier: [a-zA-Z_][a-zA-Z_0-9]*
    IdentifierStr = LastChar;
    columnNo++;

    while (isalnum((LastChar = getc(pFile))) || (LastChar == '_')) {
      IdentifierStr += LastChar;
      columnNo++;
    }

    if (IdentifierStr == "int")
      return returnTok("int", INT_TOK);
    if (IdentifierStr == "bool")
      return returnTok("bool", BOOL_TOK);
    if (IdentifierStr == "float")
      return returnTok("float", FLOAT_TOK);
    if (IdentifierStr == "void")
      return returnTok("void", VOID_TOK);
    if (IdentifierStr == "bool")
      return returnTok("bool", BOOL_TOK);
    if (IdentifierStr == "extern")
      return returnTok("extern", EXTERN);
    if (IdentifierStr == "if")
      return returnTok("if", IF);
    if (IdentifierStr == "else")
      return returnTok("else", ELSE);
    if (IdentifierStr == "while")
      return returnTok("while", WHILE);
    if (IdentifierStr == "return")
      return returnTok("return", RETURN);
    if (IdentifierStr == "true") {
      BoolVal = true;
      return returnTok("true", BOOL_LIT);
    }
    if (IdentifierStr == "false") {
      BoolVal = false;
      return returnTok("false", BOOL_LIT);
    }

    return returnTok(IdentifierStr.c_str(), IDENT);
  }

  if (LastChar == '=') {
    NextChar = getc(pFile);
    if (NextChar == '=') { // EQ: ==
      LastChar = getc(pFile);
      columnNo += 2;
      return returnTok("==", EQ);
    } else {
      LastChar = NextChar;
      columnNo++;
      return returnTok("=", ASSIGN);
    }
  }

  if (LastChar == '{') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok("{", LBRA);
  }
  if (LastChar == '}') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok("}", RBRA);
  }
  if (LastChar == '(') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok("(", LPAR);
  }
  if (LastChar == ')') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok(")", RPAR);
  }
  if (LastChar == ';') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok(";", SC);
  }
  if (LastChar == ',') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok(",", COMMA);
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9]+.
    std::string NumStr;

    if (LastChar == '.') { // Floatingpoint Number: .[0-9]+
      do {
        NumStr += LastChar;
        LastChar = getc(pFile);
        columnNo++;
      } while (isdigit(LastChar));

      FloatVal = strtof(NumStr.c_str(), nullptr);
      return returnTok(NumStr, FLOAT_LIT);
    } else {
      do { // Start of Number: [0-9]+
        NumStr += LastChar;
        LastChar = getc(pFile);
        columnNo++;
      } while (isdigit(LastChar));

      if (LastChar == '.') { // Floatingpoint Number: [0-9]+.[0-9]+)
        do {
          NumStr += LastChar;
          LastChar = getc(pFile);
          columnNo++;
        } while (isdigit(LastChar));

        FloatVal = strtof(NumStr.c_str(), nullptr);
        return returnTok(NumStr, FLOAT_LIT);
      } else { // Integer : [0-9]+
        IntVal = strtod(NumStr.c_str(), nullptr);
        return returnTok(NumStr, INT_LIT);
      }
    }
  }

  if (LastChar == '&') {
    NextChar = getc(pFile);
    if (NextChar == '&') { // AND: &&
      LastChar = getc(pFile);
      columnNo += 2;
      return returnTok("&&", AND);
    } else {
      LastChar = NextChar;
      columnNo++;
      return returnTok("&", int('&'));
    }
  }

  if (LastChar == '|') {
    NextChar = getc(pFile);
    if (NextChar == '|') { // OR: ||
      LastChar = getc(pFile);
      columnNo += 2;
      return returnTok("||", OR);
    } else {
      LastChar = NextChar;
      columnNo++;
      return returnTok("|", int('|'));
    }
  }

  if (LastChar == '!') {
    NextChar = getc(pFile);
    if (NextChar == '=') { // NE: !=
      LastChar = getc(pFile);
      columnNo += 2;
      return returnTok("!=", NE);
    } else {
      LastChar = NextChar;
      columnNo++;
      return returnTok("!", NOT);
      ;
    }
  }

  if (LastChar == '<') {
    NextChar = getc(pFile);
    if (NextChar == '=') { // LE: <=
      LastChar = getc(pFile);
      columnNo += 2;
      return returnTok("<=", LE);
    } else {
      LastChar = NextChar;
      columnNo++;
      return returnTok("<", LT);
    }
  }

  if (LastChar == '>') {
    NextChar = getc(pFile);
    if (NextChar == '=') { // GE: >=
      LastChar = getc(pFile);
      columnNo += 2;
      return returnTok(">=", GE);
    } else {
      LastChar = NextChar;
      columnNo++;
      return returnTok(">", GT);
    }
  }

  if (LastChar == '/') { // could be division or could be the start of a comment
    LastChar = getc(pFile);
    columnNo++;
    if (LastChar == '/') { // definitely a comment
      do {
        LastChar = getc(pFile);
        columnNo++;
      } while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

      if (LastChar != EOF)
        return gettok();
    } else
      return returnTok("/", DIV);
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF) {
    columnNo++;
    return returnTok("0", EOF_TOK);
  }

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  std::string s(1, ThisChar);
  LastChar = getc(pFile);
  columnNo++;
  return returnTok(s, int(ThisChar));
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/* 
* Error reporting Section */




class Warning{
  TOKEN tok;
  std::string message;

  public:
    Warning(TOKEN tok, std::string message) : tok(tok), message(message) {
      //
    }

    //\033[0;36mBE AWARE:\033[0m
    void print() const {
      std::cerr << "\033[0;33mWarning for:\033[0m '"<< tok.lexeme+"'\n" << message <<std::endl;
    }
};

void print(const TOKEN &tok, const std::string &message){
  std::cout << "Target at line " << tok.lineNo
            << " position" << tok.columnNo
            << " with value: " << tok.lexeme
            << "\n" << message << "\n";
};

void ThrowError(const TOKEN &tok, const std::string &message){
  std::cerr << "\033[0;31mError at line " << tok.lineNo
            << " with position " << tok.columnNo 
            << " token value \033[0m: " << tok.lexeme
            << "\n\033[0;33m" << message << "\033[0m\n";
  exit(1);
};



/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static TOKEN CurTok;
static std::deque<TOKEN> tok_buffer;

static TOKEN getNextToken() {

  if (tok_buffer.size() == 0)
    tok_buffer.push_back(gettok());

  TOKEN temp = tok_buffer.front();
  tok_buffer.pop_front();

  return CurTok = temp;
}

static void putBackToken(TOKEN tok) { tok_buffer.push_front(tok); }

static TOKEN lookAheadTok(size_t n = 0) {
  while (n > tok_buffer.size()) {
    tok_buffer.push_back(getNextToken());
  }
  auto iterator = tok_buffer.begin() + n;
  return *iterator;
}

//===----------------------------------------------------------------------===//
// AST nodes
//===----------------------------------------------------------------------===//

/// ASTnode - Base class for all AST nodes.
class ASTnode {
public:
  virtual ~ASTnode() {}
  virtual llvm::Value *codegen() = 0;
  virtual std::string to_string(const std::string position = "", bool atTail = false) const {return "";};
};

std::string setLevel(const std::string position, bool atTail){
      //std::cout << "Setting Level" << std::endl;

  return position + (atTail ? "└──" :"├──" );
}

std::string setPosition(const std::string position, bool terminal){
  //std::cout << "Pos set" << std::endl;

  return position + (terminal ?"" : "│   " );
}

class StmtNode : public ASTnode{
  public: StmtNode() = default;
  virtual bool isReturn() const { return false;}
  virtual std::string to_string(const std::string position="", bool atTail=true) const override { return ""; };
};

class ExprNode : public StmtNode{
  public: ExprNode() = default; 
  virtual std::string to_string(const std::string position="", bool atTail=true) const override { return ""; };
};

/* 
  * ---------------------------------------------------
  *          LLVM Helpers 
  * ---------------------------------------------------
*/

llvm::LLVMContext Context;
llvm::IRBuilder<> Builder(Context);
std::unique_ptr<llvm::Module> TheModule;


std::vector<std::unordered_map<std::string, llvm::AllocaInst*>> InScopeNames;
std::unordered_set<std::string> UnDefNames;

std::unordered_map<std::string, llvm::GlobalVariable*> Globals;

bool ifRetPath;
bool ifElsePath;
bool isBlock;

std::vector<Warning> Alerts;

llvm::Type *getRequiredPrecision(llvm::Type *expr1, llvm::Type *expr2){
    if (expr1->isFloatTy() || expr2->isFloatTy()){
        return llvm::Type::getFloatTy(Context);
    }
    else if (expr1->isIntegerTy(32) && expr2->isIntegerTy(32)){
        return llvm::Type::getInt32Ty(Context);
    }
    else if (expr1->isIntegerTy(1) || expr2->isIntegerTy(1)){
        return llvm::Type::getInt1Ty(Context);
    } else {
        ThrowError(CurTok,"ERROR: Unexpected type for expression");
        return nullptr;
    }
}
std::map<llvm::Type*, std::string> LLVMTypeStrMap{
    {llvm::Type::getVoidTy(Context), "void"},
    {llvm::Type::getInt32Ty(Context), "int"},
    {llvm::Type::getFloatTy(Context), "float"},
    {llvm::Type::getInt1Ty(Context), "bool"},
};

const std::map<TYPES, llvm::Type*> LLVMTypeMap = {
    {void_T, llvm::Type::getVoidTy(Context)},
    {int_T, llvm::Type::getInt32Ty(Context)},
    {float_T, llvm::Type::getFloatTy(Context)},
    {boolean_T, llvm::Type::getInt1Ty(Context)}
};
const std::string to_stringLLVMType(llvm::Type *PassedType){
    auto Type = LLVMTypeStrMap.find(PassedType);
    if (Type == LLVMTypeStrMap.end()){
        ThrowError(CurTok, "ERROR: Unexpected type passed. \n In MiniC it can be: 'int', 'bool', 'float' or 'void'");
    }

    return Type->second;
};
llvm::Value *Cast(llvm::Value *v, llvm::Type *t, TOKEN Tok, std::string err){
    
    if(!v){
        ThrowError(Tok, "ERROR: Invalid value for cast, v is null");
    }
    llvm::Type *vType = v->getType();
    //std::cout << "v: " << v << std::endl;

    if (vType == t){
        return v;
    }
                    // FORMAT
    //"\033[0;36m BE AWARE:\033[0m Implicit Conversion from a to b. Possible loss"+err
    
    /***    Precision Loss Section      ***/
    
    //Float to Int32
    if(t->isIntegerTy(32) && vType->isFloatTy()){
        Alerts.push_back(Warning(Tok, "\033[0;36mBE AWARE:\033[0m Implicit Conversion from Float to INT32. Possible loss"+err));
        return Builder.CreateFPToSI(v, t, Tok.lexeme.c_str());
    }

    //Float to Bool
    if(t->isIntegerTy(1) && vType->isFloatTy()){

        Alerts.push_back(Warning(Tok, "\033[0;36mBE AWARE:\033[0m Implicit Conversion from Float to Bool. Possible loss"+err));
        return Builder.CreateFCmpONE(v, llvm::ConstantFP::get(Context, llvm::APFloat(0.0)), Tok.lexeme.c_str());
    }

    //Int to Bool 
    if(t->isIntegerTy(1) && vType->isIntegerTy(32)){

        Alerts.push_back(Warning(Tok, "\033[0;36mBE AWARE:\033[0m Implicit Conversion from Int32 to Bool. Possible loss"+err));
        return Builder.CreateICmpNE(v, llvm::ConstantInt::get(Context, llvm::APInt(32, 0, true)), Tok.lexeme.c_str());
    }

    //No Precision Loss Section
    //Int32 to Float
    if(t->isFloatTy() && vType->isIntegerTy(32)){
        Alerts.push_back(Warning(Tok, "\033[0;36mBE AWARE:\033[0m Implicit Conversion from Int32 to Float."+err));
        return Builder.CreateSIToFP(v, t, Tok.lexeme.c_str());
    }

    //Bool to Float
    if(t->isFloatTy() && vType->isIntegerTy(1)){
        Alerts.push_back(Warning(Tok, "\033[0;36mBE AWARE:\033[0m Implicit Conversion from Bool to Float. Possible strange behaviour"+err));
        return Builder.CreateUIToFP(v, t, Tok.lexeme.c_str());
    }

    //Bool to Int32
    if(t->isIntegerTy(32) && vType->isIntegerTy(1)){
        Alerts.push_back(Warning(Tok, "\033[0;36mBE AWARE:\033[0m Implicit Conversion from Bool to Int32. Possible strange behaviour"+err));
        return Builder.CreateZExt(v, t, Tok.lexeme.c_str());
    }
    //std::cout << "vType: " << vType << std::endl;
    ThrowError(Tok, "ERROR: Invalid type conversion to "+to_stringLLVMType(t)+" from " + to_stringLLVMType(vType));
    return nullptr;

};


llvm::AllocaInst *CreateS1Alloca(llvm::Function *TheFunction, const std::string &VarName, llvm::Type *Type){
    llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type, nullptr, VarName.c_str());
}

llvm::Value *BoolOf(llvm::Value *v, llvm::Type *it, std::string ls, TOKEN Tok){
    if(it->isIntegerTy(1)){
        return Builder.CreateOr(v, llvm::ConstantInt::get(Context, llvm::APInt(1, 0, true)), ls);
    }
    if(it->isIntegerTy(32)){
        return Builder.CreateICmpNE(v, llvm::ConstantInt::get(Context, llvm::APInt(32, 0, true)), ls);
    }
    if(it->isFloatTy()){
        return Builder.CreateFCmpONE(v, llvm::ConstantFP::get(Context, llvm::APFloat(0.0)), ls);
    }
    ThrowError(Tok,"SEMANTIC ERROR: Void types cannot be in conditional expressions");

    return nullptr;
}

llvm::Type *TTLLVM(TYPES PassedType, TOKEN Tok){
    auto Type = LLVMTypeMap.find(PassedType);
    if(Type == LLVMTypeMap.end()){
        ThrowError(Tok, "ERROR: Unexpected type passed. \n In MiniC it can be: 'int', 'bool', 'float' or 'void'");
    }
    return Type->second;
};




static AllocaInst* CreateEntryBlockAlloca(Function *TheFunction, const std::string &VarName, Type *Type){
    IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type, nullptr, VarName.c_str());
};



llvm::Value *LazyOr(TOKEN operate, std::unique_ptr<ExprNode> L, std::unique_ptr<ExprNode> R){
    llvm::Function *TheFunction = Builder.GetInsertBlock()->getParent();
    //llvm::AllocaInst *tAlloca = CreateEntryBlockAlloca(TheFunction, "LazyOr", llvm::Type::getInt1Ty(Context));
    
    llvm::BasicBlock *RL = llvm::BasicBlock::Create(Context, "RSide", TheFunction);
    llvm::BasicBlock *REnd = llvm::BasicBlock::Create(Context, "EndExpr");

    llvm::Value *RBool;
    llvm::Value *LAssign = L->codegen();

    llvm::Type *LType = LAssign->getType();

    if(LType->isVoidTy()){
        ThrowError(operate,"ERROR: Cannot have void type in expression");
    }

    llvm::PHINode *tempAllocation = Builder.CreatePHI(llvm::Type::getInt1Ty(Context), 2, "tmplazyand");

    llvm::Value *LCast = Cast(LAssign, llvm::Type::getInt1Ty(Context), operate, " in LazyOr");
    
    Builder.CreateCondBr(LCast, REnd, RL);

    Builder.SetInsertPoint(RL);

    llvm::Value *RAssign = R->codegen();
    
    llvm::Type *RType = RAssign->getType();
    if(RType->isVoidTy()){
        ThrowError(operate, "ERROR: Void type in RHS of expression");
    }

    RBool = Cast(RAssign, llvm::Type::getInt1Ty(Context), operate, "");
    Builder.CreateBr(REnd);

    REnd->insertInto(TheFunction);
    Builder.SetInsertPoint(REnd);

    tempAllocation->addIncoming(RBool,RL);
    tempAllocation->addIncoming(LCast,Builder.GetInsertBlock());

    return tempAllocation;

};

llvm::Value *LazyAnd(TOKEN operate, std::unique_ptr<ExprNode> L, std::unique_ptr<ExprNode> R){
    llvm::Function *TheFunction = Builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *RL = llvm::BasicBlock::Create(Context, "RSide", TheFunction);
    llvm::BasicBlock *REnd = llvm::BasicBlock::Create(Context, "EndExpr");
    //llvm::BasicBlock *Continue = llvm::BasicBlock::Create(Context, "Cont");

    llvm::Value *RBool;
    llvm::Value *LAssign = L->codegen();
    llvm::Type *LType = LAssign->getType();

    if(LType->isVoidTy()){
        ThrowError(operate, "ERROR: Cannot have void type in expression");
    }

    llvm::PHINode *tempAllocation = Builder.CreatePHI(llvm::Type::getInt1Ty(Context), 2, "tmplazyand");
    
    llvm::Value *LCast = Cast(LAssign, llvm::Type::getInt1Ty(Context), operate, " in LazyAnd");
    
    Builder.CreateCondBr(LCast, RL, REnd); //Branch to RL if L is true
    
    Builder.SetInsertPoint(RL);

    llvm::Value *RAssign = R->codegen();
    llvm::Type *RType = RAssign->getType();
    if(RType->isVoidTy()){
        ThrowError(operate, "ERROR: Void type in RHS of expression");
    }

    RBool = Cast(RAssign, llvm::Type::getInt1Ty(Context), operate, "");
    Builder.CreateBr(REnd);
    //Branch - skip RSide

    REnd->insertInto(TheFunction);
    Builder.SetInsertPoint(REnd);

    tempAllocation->addIncoming(RBool,RL);
    tempAllocation->addIncoming(LCast,Builder.GetInsertBlock());

    return tempAllocation;
};

static llvm::Value *asBool(llvm::Value *v, llvm::Type *t, std::string rec){
  
  if(t->isFloatTy()){
    return Builder.CreateFCmpONE(v, llvm::ConstantFP::get(Context, llvm::APFloat(0.0)), rec);
  } else if (t->isIntegerTy(1)){
    //std::cout << "Bool" << std::endl;
    return Builder.CreateICmpNE(v, llvm::ConstantInt::get(Context, llvm::APInt(1, 0)), rec);
  } else if (t->isIntegerTy(32)){
    return Builder.CreateICmpNE(v, llvm::ConstantInt::get(Context, llvm::APInt(32, 0)), rec);
    // true or false = signed/unsigned
  } else {
    ThrowError(CurTok, "ERROR: Void types cannot be in conditional expressions"); 

  }
  return nullptr;

};



// ---------------------------------------------------
//      -------------END OF HELPERS--------------
// ---------------------------------------------------




const std::map<std::string, TYPES> TypeMap = {
  {"void", void_T},
  {"int", int_T},
  {"float", float_T},
  {"bool", boolean_T}
};

const std::map<TYPES, std::string> RevTypeMap = {
  {void_T, "void"},
  {int_T, "int"},
  {float_T, "float"},
  {boolean_T, "bool"}
};


/*

  * CLASS ONE

*/

class ParamNode : public ASTnode{
  TYPES Type;
  TOKEN id;

  public: ParamNode(TYPES Type, TOKEN id) : Type(std::move(Type)), id(std::move(id)) {
    //
  }

  virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    //std::cout << "Param= " << id.lexeme << " with type: " << RevTypeMap.at(Type) << std::endl;
    return setLevel(position, atTail) + "Param='" + id.lexeme + "' with type: '" + RevTypeMap.at(Type) + "'\n";
  }

  virtual llvm::Value *codegen() override { return nullptr; };

  TYPES getType() const { return Type; }
  TOKEN getName() const  { return id; }

};

class VarDecNode : public ASTnode{
  TYPES Type;
  TOKEN initier;

  public: VarDecNode(TYPES type, TOKEN initier) :  
    Type(std::move(type)), initier(std::move(initier)) {
  }
  virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    //std::cout<< "VarDec= " << initier.lexeme << " with type: " << RevTypeMap.at(Type) << std::endl;
    std::string lvl = setLevel(position, atTail);
    std::string temp =  lvl + "VarDec='" + initier.lexeme +"'";
    temp += "with type='"+RevTypeMap.at(Type) +"'\n";
    //std::cout<< temp << std::endl;
    return temp;
  }

  virtual Value *codegen() override { 
    // Global or Scoped = T or F respectively
    bool fullScope = InScopeNames.size() == 0;
    llvm::Type *llvmType = TTLLVM(Type, initier);

    // Switch used for better readability w/ way variable is named
    //std::cout << "Vardecl IR"<< std::endl;
    if(fullScope){
      //true:{ //Global Variable
        //std::cout << "Global Variable" << std::endl;
        if(Globals.find(initier.lexeme) != Globals.end()){
          ThrowError(initier,"ERROR: Variable already declared in global scope");
        }
                   // Apparently CommonLinkage is better than External Linkage (It is more flexible)
        auto Glob = new llvm::GlobalVariable(*(TheModule.get())
        , llvmType, false, llvm::GlobalValue::CommonLinkage,llvm::Constant::getNullValue(llvmType) ,initier.lexeme);          
        
        UnDefNames.insert(initier.lexeme);
        //std::cout << "Global Variable insert " << initier.lexeme <<std::endl;
        Globals.insert({initier.lexeme, Glob});
        
    } else{ //Local Variable 
        //std::cout << "Local Variable" << std::endl;
        if(InScopeNames.back().find(initier.lexeme) != InScopeNames.back().end()){
            ThrowError(initier, "ERROR: Variable already declared in local scope");
        }
        llvm::Function *TheFunction = Builder.GetInsertBlock()->getParent();
        llvm::AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, initier.lexeme, llvmType);

        Builder.CreateStore(llvm::Constant::getNullValue(llvmType), Alloca);
        UnDefNames.insert(initier.lexeme);
        InScopeNames.back().insert({initier.lexeme, Alloca});
      }
    
    return nullptr; 
  };


};


class ReturnNode : public StmtNode{
  std::unique_ptr<ExprNode> RetExpr;
  TOKEN Tok;

  public:
    ReturnNode(std::unique_ptr<ExprNode> RetExpr, TOKEN Tok) 
    : RetExpr(std::move(RetExpr)), Tok(Tok)
      {
        //
      }
      virtual std::string to_string(const std::string position="",bool atTail=true) const override { 
          std::string temp = setLevel(position, atTail);
          if (RetExpr){
            temp += "Return with value:\n";
            std::string nextPos = setPosition(position, false);
            temp +=  RetExpr->to_string(nextPos,true);
            //temp += "\n";
          } else{
            temp += "Return\n";
          }
          return temp;
      }
      
    TOKEN getTok() const { return Tok; }

  bool isReturn() const override { return true; }

  virtual Value *codegen() override {
    //llvm::Function *TheFunction = Builder.GetInsertBlock()->getParent()->getReturnType();
    llvm::Function *TheFunction = Builder.GetInsertBlock()->getParent();
    llvm::Type *ReType = TheFunction->getReturnType();

    if(RetExpr != nullptr){
        //std::cout << RetExpr->to_string() << std::endl;
        llvm::Value *RetValue = RetExpr->codegen();
        llvm::Type *ValType = RetValue->getType();
        //std::cout << "ValType: " << ValType << std::endl;
        //std::cout << "ReType: " << ReType << std::endl;
        if(ValType != ReType){

            ThrowError(Tok, "ERROR: Mismatch of return type and function types. Expected: " + to_stringLLVMType(ReType) + ", but got: " + to_stringLLVMType(ValType));
        }
        Builder.CreateRet(RetValue);
    } else{
        if(ReType->isVoidTy()){
            Builder.CreateRetVoid();
            return nullptr;
        } else{
            ThrowError(Tok, "ERROR: Expected return value in non-void function");
        }        
    }
    return nullptr;
  }

};

class BlockNode : public StmtNode{
  std::vector<std::unique_ptr<VarDecNode>> LocalVDecs;
  std::vector<std::unique_ptr<StmtNode>> LocalStmts;

  public: BlockNode(std::vector<std::unique_ptr<VarDecNode>> LocalVDecs, 
  std::vector<std::unique_ptr<StmtNode>> LocalStmts) : LocalVDecs(std::move(LocalVDecs)),
  LocalStmts(std::move(LocalStmts)) {
    //
  }

  virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    std::string nextpos = setPosition(position, atTail); 
    std::string temp;
    //std::cout << "Block" << std::endl;
    for (size_t i = 0; i < LocalVDecs.size(); i++) {
        bool isLast = (i == LocalVDecs.size() - 1) && LocalStmts.empty();
        temp += LocalVDecs[i]->to_string(nextpos, isLast);
    }
    
    // Process statements
    for (size_t i = 0; i < LocalStmts.size(); i++) {
        if (LocalStmts[i] == nullptr) {
            std::cerr << "stmt is null" << std::endl;
            continue;
        }
        bool isLast = (i == LocalStmts.size() - 1);
        temp += LocalStmts[i]->to_string(nextpos, isLast);
    }
    
    return temp;
  }

  virtual Value *codegen() override { 
    bool newScope = !isBlock;

    // Essentially, is there a return statement in the block
    llvm::Value *isProcedure = nullptr;


    if(newScope){
        InScopeNames.push_back(std::unordered_map<std::string, llvm::AllocaInst*>());
    } else{
        isBlock = false;
    }

    for (auto &LocalVDec : LocalVDecs){
        LocalVDec->codegen();
    }

    int listIdTicker = 0;
    for (auto &LocalStmt : LocalStmts){
      /* dyn_cast<ReturnNode> */
        
        if(LocalStmt->isReturn()){
            auto returnBlock = static_cast<ReturnNode*>(LocalStmt.get()); // check 
            if (listIdTicker != LocalStmts.size()-1){
                int yPos = returnBlock->getTok().lineNo;
                int xPos = returnBlock->getTok().columnNo;
                Alerts.push_back(Warning(returnBlock->getTok(), "\033[0;36mBE AWARE: \033[0m Return statement in block at line "+std::to_string(yPos)+" column "+std::to_string(xPos)+"\033[0m"));
            }
            isProcedure = LocalStmt->codegen();

            if(newScope){
                InScopeNames.pop_back();
            }
            return isProcedure;

        } else{
            isProcedure = LocalStmt->codegen();
        }   
    }
    if(newScope){
        InScopeNames.pop_back();
    }
    if(isProcedure){
        if(isProcedure->getType()->isVoidTy()){
            return isProcedure;
        }
    }
    return nullptr;  
  }

};

class FuncDecNode : public ASTnode{
  std::vector<std::unique_ptr<ParamNode>> Params;
  TYPES Type;
  TOKEN id;

  //Non Extern 
  std::unique_ptr<BlockNode> Body;

  public: FuncDecNode(TYPES type, TOKEN id, std::vector<std::unique_ptr<ParamNode>> params, std::unique_ptr<BlockNode> body)
    : Type(std::move(type)), id(std::move(id)), Params(std::move(params)), Body(std::move(body)) {
      //
    }

    virtual std::string to_string(const std::string position="", bool atTail=false) const override {
    std::string temp = setLevel(position, atTail);
    temp += Body ? "FuncDec: type='" : "ExternDef: type='";
    temp += RevTypeMap.at(Type) + "' name: '" + id.lexeme + "'\n";
    
    std::string nextPos = setPosition(position, false);
    
    for (size_t i = 0; i < Params.size(); i++) {
        bool isLast = (i == Params.size()-1) && (Body == nullptr);
        temp += Params[i]->to_string(nextPos, isLast);
    }
    
    if(Body != nullptr){
        temp += Body->to_string(nextPos, true);
    }
    return temp;
}

    virtual Value *codegen() override {
      std::cout << "Codegen FuncDecNode" << std::endl; 
      std::cout << "Function name: " << id.lexeme << std::endl;

      llvm::FunctionType *retFType;
      llvm::Function *Fdef;
      std::vector<llvm::Type*> args;
      std::unordered_set<std::string> argNames;


      bool isProcedure = true;

      Function *externF = TheModule->getFunction(id.lexeme);

      //std::cout << "  stop 1"  << std::endl;

      if(!externF){
        for (auto &each: Params){
          args.push_back(TTLLVM(each->getType(), id));
        }

        if(args.empty()){
          retFType = llvm::FunctionType::get(TTLLVM(Type, id), false);
        } else {
          retFType = llvm::FunctionType::get(TTLLVM(Type, id), args, false);
        }
        Fdef = llvm::Function::Create(retFType, llvm::Function::ExternalLinkage, id.lexeme, TheModule.get());
        
        
        int ticker = 0;
        for(auto &eachFunArg : Fdef->args()){
          std::string cargName = Params[ticker++]->getName().lexeme;
          
          if(argNames.find(cargName) != argNames.end()){
            ThrowError(id, "ERROR: Duplicate argument name");
          }
          argNames.insert(cargName);
          eachFunArg.setName(cargName);
        }

        if(!Body){ //Extern
          return Fdef;
        }

      }

    //std::cout << "  stop 2"  << std::endl;

      llvm::BasicBlock *BB = llvm::BasicBlock::Create(Context, "Func entry", Fdef);
      Builder.SetInsertPoint(BB);

      //std::cout << "BBDone" << std::endl;

      std::unordered_map<std::string, llvm::AllocaInst*> NewScope;
      InScopeNames.push_back(NewScope);
      //std::cout << " pushed back"<< std::endl;

      for(auto &Arg : Fdef->args()){
        llvm::AllocaInst *Alloca = CreateEntryBlockAlloca(Fdef, Arg.getName().data(), Arg.getType());

        //std::cout << "  stop 2.5"  << std::endl;
        //std::cout << Arg.getName().data() << std::endl;

        Builder.CreateStore(&Arg, Alloca);
        InScopeNames.back()[Arg.getName().data()] = Alloca;
        //std::cout << "  stop 2.6"  << std::endl;
      }

      isBlock = true;
      isProcedure = Body->codegen() == nullptr;
      //std::cout << isProcedure  << std::endl;


      //std::cout << "  stop 3"  << std::endl;

      bool allPaths = ifRetPath || isProcedure;
      llvm::Type *curType = Fdef->getReturnType();
    


      if(!allPaths){
        if(curType->isVoidTy()){
          Builder.CreateRetVoid();
        } else {
          ThrowError(id, "ERROR: All paths must return a value");
        }
      }
      //std::cout << "  stop 5"  << std::endl;
      if(allPaths && !isProcedure){
        Alerts.push_back(Warning(id, "\033[0;36mWARNING: \033[0mCode after conditional statements may be unreachable"));

        if(curType->isVoidTy()){
          Builder.CreateRetVoid();
       } else if(curType->isIntegerTy(1)){
          Builder.CreateRet(llvm::ConstantInt::get(Context, llvm::APInt(1, 0)));
        } else if(curType->isIntegerTy(32)){
          Builder.CreateRet(llvm::ConstantInt::get(Context, llvm::APInt(32, 0, true)));
        } else if(curType->isFloatTy()){
          Builder.CreateRet(llvm::ConstantFP::get(Context, llvm::APFloat(0.0)));
        } else {
          ThrowError(id, "ERROR: All paths must return a value");
        }
    
      }
    //std::cout << "  stop 6"  << std::endl;
      llvm::verifyFunction(*Fdef);
      InScopeNames.pop_back();

      return Fdef;
    }
};

class DeclNode : public ASTnode{
  std::unique_ptr<FuncDecNode> FuncDec;
  std::unique_ptr<VarDecNode> VarDec;

  public: DeclNode(std::unique_ptr<FuncDecNode> funcDec, std::unique_ptr<VarDecNode> varDec)
    : FuncDec(std::move(funcDec)), VarDec(std::move(varDec)) {
    //
  }

virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    if (FuncDec) {
        return FuncDec->to_string(position, atTail);
    }
    if (VarDec) {
        return VarDec->to_string(position, atTail);
    }
    return setLevel(position, atTail) + "empty\n";
}

  virtual Value *codegen() override { 
    if(FuncDec){
        FuncDec->codegen();
    }
    if(VarDec){
        VarDec->codegen();
    }

    //std::cout << "Codegen DeclNode done" << std::endl;

    return nullptr;
  }

};

class ProgramNode : public ASTnode{
  std::vector<std::unique_ptr<FuncDecNode>> Externs;
  std::vector<std::unique_ptr<DeclNode>> Decls;

  public: ProgramNode(std::vector<std::unique_ptr<FuncDecNode>> Externs, std::vector<std::unique_ptr<DeclNode>> Decls)
  : Externs(std::move(Externs)), Decls(std::move(Decls)) {}

  virtual std::string to_string(const std::string position="", bool atTail=false) const override {
    std::string temp = setLevel(position, true) + "Program\n";
    std::string nextpos = setPosition(position, atTail);
    
    size_t total = Externs.size() + Decls.size();
    size_t count = 0;
    
    for(const auto& ext : Externs) {
        temp += ext->to_string(nextpos, ++count == total);
    }
    
    for(const auto& decl : Decls) {
        temp += decl->to_string(nextpos, ++count == total);
    }
    
    return temp;
  }

  virtual Value *codegen() override {
    //std::cout << "Codegen Program" << std::endl;

    for( auto &Extern : Externs){
      //std::cout << "Extern" << std::endl;
        Extern->codegen();
    }
    for( auto &Decl : Decls){
      //std::cout << "Decl" << std::endl;
      Decl->codegen(); 

    }
    return nullptr;  
  };
};
/*class ExternNode : public ASTnode{
  */ 
//Don't need an ExternNode class because it is just a function declaration


/*

  * CLASS 2

*/

/// IntASTnode - Class for integer literals like 1, 2, 10,
class IntNode : public ExprNode {
  std::string Val;
  TOKEN Tok;
  //std::string Name;

public:
  IntNode(TOKEN tok) : Val(tok.lexeme), Tok(tok) {
    //
  }

  virtual std::string to_string(const std:: string position="", bool atTail = true) const override {
    std::string valStr = "IntLIT='" + Val + "'\n";
    std::string lvl = setLevel(position, true);
    return lvl + valStr;
  }

  virtual Value *codegen() override { 
      //Val = (int)Val;
    return ConstantInt::get(Context, APInt(32, std::stoi(Val), true)); }
                          // CHECK -> the true, why?
  
  // virtual std::string to_string() const override {
  // return a sting representation of this AST node
  //};
};

class FloatLitNode : public ExprNode{
  std::string Val;
  TOKEN Tok;
  //std::string Name;

public:
  FloatLitNode(TOKEN tok) : Val(tok.lexeme), Tok(tok){
    //
  }

  virtual std::string to_string(const std:: string position ="", bool atTail = true) const override {
    std::string valStr = "FloatLIT= " + Val + "\n";
    std::string lvl = setLevel(position, true);
    return lvl+valStr;
  }
  //toString
  llvm::Value *codegen() override{
    return ConstantFP::get(Context, llvm::APFloat(stof(Val)));
  }
  // maybe APFloat::IEEEsingle()
};

class BoolLitNode : public ExprNode{
  std::string Val;
  TOKEN Tok;
  //std:: string Name

  public: BoolLitNode(TOKEN Tok) : Tok(Tok), Val(Tok.lexeme) {
      //
    }

    virtual std:: string to_string(const std:: string position ="", bool atTail = true) const override {
      std::string valStr = "BoolLIT='" + Val + "'\n";
      std::string lvl = setLevel(position, true);
      return lvl+valStr;
    }

  virtual Value *codegen() override{
    // int or float
    bool heldValue = (Val == "true" ? 1 : 0);
    return llvm::ConstantInt::get(llvm::Type::getInt1Ty(Context), heldValue);
  }
};


/*

 * CLASS THREE

*/


class BinaryExprNode : public ExprNode {
  std::unique_ptr<ExprNode> LHS;
  std::unique_ptr<ExprNode> RHS;
  TOKEN operate;

  public: BinaryExprNode(std::unique_ptr<ExprNode> lhs, std::unique_ptr<ExprNode> rhs, TOKEN operate)
    : LHS(std::move(lhs)), RHS(std::move(rhs)), operate(operate) {
      //
    }

    virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    std::string temp = setLevel(position, atTail);
    temp += "BinaryExpr:\n";
    std::string nextPos = setPosition(position, false);
    
    if (LHS) {
        temp += LHS->to_string(nextPos, false);  // LHS is not last
    }
    
    // Operator as a middle child
    temp += setLevel(nextPos, false) + "Operator='" + operate.lexeme + "'\n";
    
    if (RHS) {
        temp += RHS->to_string(nextPos, true);  // RHS is last
    }
    
    return temp;
}

    virtual Value *codegen() override {
      llvm::Value *L;
      llvm::Value *R ;
      llvm::Value *Result;
      //llvm::Type *Type;
      llvm::Type *requiredPrecision;

      //std::cout << "BinaryExprNode part 1" << std::endl;
      

      switch(operate.type){
          case PLUS:
          case MINUS:
          case ASTERIX:
          case DIV:
          case GT:
          case LT:
          case GE:
          case LE:
          case NE:
          case EQ:
          case MOD:
          {
            //std::cout << "BinaryExprNode part 1.5" << std::endl;
            //std::cout << operate.lexeme << std::endl;
            llvm::Value *LHV = LHS->codegen();
            //std::cout << "LHV" << std::endl;
            llvm::Value *RHV = RHS->codegen();
            //std::cout << "RHV" << std::endl;
            if(LHV->getType()->isVoidTy() || RHV->getType()->isVoidTy()){
              ThrowError(operate, "ERROR: Cannot have void type in expression");
            }
            //std::cout << "BinaryExprNode part 1.6" << std::endl;
            requiredPrecision = getRequiredPrecision(LHV->getType(), RHV->getType());
            
            //std::cout << "require" << std::endl;
            //std::cout << requiredPrecision << std::endl;
            //std::cout << operate.lexeme << std::endl;
            L = Cast(LHV, requiredPrecision, operate, " on the lhs of the operator");
            R = Cast(RHV, requiredPrecision, operate, " on the rhs of the operator");
          break;
          }
          case AND:
              return LazyAnd(operate, std::move(LHS), std::move(RHS));
          case OR:
              return LazyOr(operate, std::move(LHS), std::move(RHS));
          default:
              ThrowError(operate, "ERROR: Invalid operator for Binary expression");       
      }

      //std::cout << "BinaryExprNode part 2" << std::endl;
      //std::cout << operate.lexeme << std::endl;
      //std::cout << operate.type << std::endl;
      //std::cout << R << std::endl;
      //std::cout << requiredPrecision << std::endl;


    switch(operate.type){
        case PLUS:{
          if (requiredPrecision->isFloatTy()){
            Result = Builder.CreateFAdd(L, R, "faddtmp");
          } else {
            Result = Builder.CreateAdd(L, R, "addtmp");
          }
          return Result;
        }
        case MINUS:{
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFSub(L, R, "fsubtmp");
          } else {
              Result = Builder.CreateSub(L,R, "isubtmp");
          }
          return Result;
        }
        case ASTERIX:{
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFMul(L, R, "fmultmp");
          } else{
              Result = Builder.CreateMul(L, R, "multmp");
          }
          return Result;
        }
        case DIV:{
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFDiv(L, R, "fdivtmp");
          } else if (requiredPrecision->isIntegerTy(32)){
              Result = Builder.CreateSDiv(L, R, "idivtmp");
          } else{
              Result = Builder.CreateUDiv(L, R, "bdivtmp");
          }
          return Result;
        }
        case MOD:{
          if(requiredPrecision->isFloatTy()){
              //ThrowError(operate, "ERROR: Modulus operate cannot be used on float");
              Result = Builder.CreateFRem(L, R, "fmodtmp");
          } else if (requiredPrecision->isIntegerTy(1)){
              Result = Builder.CreateURem(L, R, "bmodtmp");
          } else{
              Result = Builder.CreateSRem(L, R, "modtmp");
          }
          return Result;
        }
        case GT:{
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFCmpOGT(L, R, "fcmpgttmp");
          } else{
              Result = Builder.CreateICmpSGT(L, R, "icmpgttmp");
          }
          return Result;
        }
        case LT:{
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFCmpOLT(L, R, "fLTtmp");
          } else if (requiredPrecision->isIntegerTy(1)){ 
              Result = Builder.CreateICmpULT(L, R, "bLTtmp");
          } else if (requiredPrecision->isIntegerTy(32)){
              Result = Builder.CreateICmpSLT(L, R, "iLTtmp");
          }
          return Result;
        }
        case EQ:{
          if (requiredPrecision->isFloatTy()){
              Result = Builder.CreateFCmpOEQ(L, R, "fcmpeqtmp");
          } else {
              Result = Builder.CreateICmpEQ(L, R, "icmpeqtmp");
          }
          return Result;
        }
        case LE:{
          //std::cout << "LE" << std::endl;
          //std::cout << requiredPrecision->isFloatTy() << std::endl;
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFCmpOLE(L, R, "fcmpletmp");
          } else if(requiredPrecision->isIntegerTy(1)){
              Result = Builder.CreateICmpULE(L, R, "icmpetmp");
          } else if (requiredPrecision->isIntegerTy(32)){
              Result = Builder.CreateICmpSLE(L, R, "icmpetmp");
          }
          return Result;
        }
        case GE:{
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFCmpOGE(L, R, "fcmpgetmp");
          } else{
              Result = Builder.CreateICmpSGE(L, R, "icmpgetmp");
          }
          return Result;
        }
        case NE:{
          if(requiredPrecision->isFloatTy()){
              Result = Builder.CreateFCmpONE(L, R, "fcmpnetmp");
          } else{
              //std::cout << "NE" << std::endl;
              Result = Builder.CreateICmpNE(L, R, "icmpnetmp");
          }
          return Result;
        }
    }
    return nullptr;
  }

};

class UnaryNode : public ExprNode{
  std::unique_ptr<ExprNode> Expr;
  TOKEN operate;

  public: UnaryNode(std::unique_ptr<ExprNode> expr, TOKEN op) : Expr(std::move(expr)), operate(op) {
    //
  }

  virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    std::string temp = setLevel(position, atTail);
    temp += "UnaryOp='" + operate.lexeme + "'\n";
    if (Expr){
      temp += Expr->to_string(setPosition(position, false), true);
    }
    return temp;
  }

  virtual Value *codegen() override {
    llvm::Value *ExprVal = Expr-> codegen();
    llvm::Type *ExprType = ExprVal->getType();
    llvm::Value *UnExpr;
    llvm::Value *casted = ExprVal;

    if(ExprType->isVoidTy()){
        ThrowError(operate, "ERROR: Cannot have void type in expression");
    }

    switch(operate.type){
        case MINUS:
            if (ExprType->isIntegerTy()){
                UnExpr = Builder.CreateNeg(ExprVal, "negint");
            
            } else if (ExprType->isFloatTy()){
                UnExpr = Builder.CreateFNeg(ExprVal, "negflo");
            } else{
                    ThrowError(operate, "ERROR: Invalid type for Negative unary expression");
            }
            break;
        case NOT:
            if(ExprType->isIntegerTy(1)){
                UnExpr = Builder.CreateXor(ExprVal, ConstantInt::get(Context, APInt(1, 1)), "nottmp");
            } else {
                UnExpr = Builder.CreateNot(ExprVal, "nottmp");
            }
            break;
        default:
            ThrowError(operate, "ERROR: Invalid operate for Unary expression");
            
    }
    return UnExpr; }
};

/*

 * CLASS FOUR 


*/



class VarValueNode : public ExprNode{
  TOKEN id;

  public: VarValueNode(TOKEN id) : id(std::move(id)) {
    //
  }

  virtual std::string to_string(const std::string position="", bool atTail=true) const override {    
    std::string valStr = "Variable= '" + id.lexeme + "'\n";
    std::string lvl = setLevel(position, atTail);
    return lvl + valStr;
    
  }

  virtual Value *codegen() override { 

    //std::cout << "VarValueNode" << std::endl;
    //std::cout << id.lexeme << std::endl;
    //llvm::AllocaInst *retval = nullptr;


    for (int i = InScopeNames.size()-1; i >= 0; i--){
      if(InScopeNames[i].find(id.lexeme) != InScopeNames[i].end()){
        llvm::AllocaInst *retval = InScopeNames[i].at(id.lexeme);
        if(UnDefNames.count(id.lexeme)!=0){
          UnDefNames.erase(id.lexeme);
          Alerts.push_back(Warning(id,"\033[0;36m WARNING: \033[0mVariable used before being defined, default taken."));
        }
        llvm::Type *AllocType = retval->getAllocatedType();
        
        //std::cout << "VarValueNode breaks here" << std::endl;
        //std::cout << id.lexeme << std::endl;
                
        //std::cout << retval << std::endl;
        //std::cout << "VarValueNode part 2" << std::endl;
        llvm::Value *loadedD = Builder.CreateLoad(AllocType, retval, id.lexeme.c_str());
        

        return loadedD;
      }
    }

    //std::cout << "VarValueNode part 3" << std::endl;

    for (int i = Globals.size()-1; i >=0;i--){
      if (Globals.find(id.lexeme) != Globals.end()){
        llvm::GlobalVariable *Glob = Globals.at(id.lexeme);
        if(UnDefNames.count(id.lexeme)!=0){
          UnDefNames.erase(id.lexeme);
          Alerts.push_back(Warning(id,"\033[0;36m WARNING: Variable used before being defined"));
        }
      return Builder.CreateLoad(Glob->getValueType(), Glob, id.lexeme);
      }
    }
    ThrowError(id, "ERROR: Variable not declared in scope or globally");
    return nullptr;
  
  };

  const TOKEN& getTok() const { return id; } ;
};

class VarCallNode : public ExprNode{
  TOKEN id;
  std::unique_ptr<ExprNode> variableCall;

  public: VarCallNode(TOKEN id, std::unique_ptr<ExprNode> variableCall) : id(std::move(id)), variableCall(std::move(variableCall)){
    //
  }

virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    std::string temp = setLevel(position, atTail);
    temp += "VarCall for:'" + id.lexeme + "'\n";
    if (variableCall != nullptr){
        std::string nextPos = setPosition(position, false);
        temp += variableCall->to_string(nextPos, true);
    }
    return temp;
}

virtual Value *codegen() override { 
  llvm::Value *expression = variableCall->codegen();
  if(expression->getType()->isVoidTy()){
    ThrowError(id, "ERROR: Variable calls cannot have void type");
  }
  for (int i = InScopeNames.size()-1; i >= 0; i--){
    if(InScopeNames[i].find(id.lexeme) != InScopeNames[i].end()){
      
      llvm::AllocaInst *retval = InScopeNames[i].at(id.lexeme);
      llvm::Value *casted = Cast(expression, retval->getAllocatedType(), id, " in VarCall");
      Builder.CreateStore(casted, retval);
      UnDefNames.erase(id.lexeme);
      return casted;
      // Builder.CreateLoad(retval->getAllocatedType(), retval, id.lexeme.c_str());
    }
  }
  if(Globals.find(id.lexeme) != Globals.end()){
    GlobalVariable *Glob = Globals.at(id.lexeme);
    llvm::Value *casted = Cast(expression, Glob->getValueType(), id, " in VarCall");
    Builder.CreateStore(casted, Glob);
    Globals[id.lexeme] = Glob;
    UnDefNames.erase(id.lexeme);
    return Builder.CreateLoad(Glob->getValueType(), Glob, id.lexeme.c_str());
    } else{
      ThrowError(id, "ERROR: Variable not declared in scope or globally before call");
    }
  return nullptr; // won't be reached, just avoids warnings
  }
};

/*

 * CLASS FIVE
 
*/


class FuncCallNode : public ExprNode{
  std::vector<std::unique_ptr<ExprNode>> Args;
  TOKEN Funcid;

  public: FuncCallNode(TOKEN Funcid, std::vector<std::unique_ptr<ExprNode>> Args)
    : Funcid(std::move(Funcid)), Args(std::move(Args)) {
      //
    }

    virtual std::string to_string(const std::string position="", bool atTail=true) const override {
    std::string temp = setLevel(position, atTail);  // Use atTail, not false
    temp += "Function call for: '" + Funcid.lexeme + "'\n";
    
    if (!Args.empty()) {
        std::string nextPos = setPosition(position, false);
        temp += setLevel(nextPos, true) + "Arg(s)\n";  // Args node is last child
        
        std::string argsPos = setPosition(nextPos, false);  // Position for arg children
        
        for(size_t i = 0; i < Args.size(); i++) {
            temp += Args[i]->to_string(argsPos, i == Args.size()-1);
        }
    }
    
    return temp;
  }

    virtual Value *codegen() override {
        llvm::Function *Caller = TheModule->getFunction(Funcid.lexeme);

        if (!Caller){
          ThrowError(Funcid, "ERROR: Function call recipient not found");
        }
        /*if(Args.size() != Caller->arg_size()){
          ThrowError(Funcid, "ERROR: Function call argument size mismatch, required size is " + std::to_string(Caller->arg_size()));
        }*/

        llvm::Type *FArgType;
        std::vector<llvm::Value*> ArgsVals;

        int argID=0;
        for (auto &eachArg : Args){
          FArgType = Caller->getArg(argID)->getType();
          llvm::Value *ArgVal = eachArg->codegen();
          llvm::Type *ArgType = ArgVal->getType();
          ArgsVals.push_back(Cast(ArgVal, FArgType, Funcid, " in FuncCall"));
          argID++;
        }
      return Builder.CreateCall(Caller, ArgsVals, Funcid.lexeme);

    };

};




/*

 * CLASS SIX

*/
class WhileNode : public StmtNode {
  // type Val;
  TOKEN Tok;
  //std::string Name;
  std::unique_ptr<ExprNode> Cond;
  std::unique_ptr<StmtNode> LoopSection;


  public: WhileNode(TOKEN tok, std::unique_ptr<ExprNode> Cond, std::unique_ptr<StmtNode> loopSection)
    : Tok(tok), Cond(std::move(Cond)), LoopSection(std::move(loopSection)) {
      //
    }

    virtual std::string to_string(const std::string position="", bool atTail=true) const override {
      std::string temp = setLevel(position, atTail);
      temp += "While\n";
      if (Cond){
        temp += Cond->to_string(setPosition(position, atTail), false);
      }
      if (LoopSection){
        temp += LoopSection->to_string(setPosition(position,atTail), true);
      }
      return temp;
    }

  

    virtual Value *codegen() override { 
    llvm::Value *CondVal;
    llvm::Value *CondBoolVal;
    
    llvm::Function *TheFunction = Builder.GetInsertBlock()->getParent();
    
    llvm::BasicBlock *CondBB = llvm::BasicBlock::Create(Context, "condwhile", TheFunction);

    llvm::BasicBlock *LoopBB = llvm::BasicBlock::Create(Context, "loopwhile", TheFunction);
    llvm::BasicBlock *AfterBB = llvm::BasicBlock::Create(Context, "afterwhile",TheFunction);

    Builder.CreateBr(CondBB);

    Builder.SetInsertPoint(CondBB);


    CondVal = Cond->codegen();
    //std::cout << Cond->to_string() << std::endl;
    //std::cout << "WhileNode part 2" << std::endl;
    if(!CondVal){
      std::cerr << "CondVal is null" << std::endl;
    }
    //std::cout << CondVal << std::endl;
    llvm::Type *CondVT = CondVal->getType();
    CondBoolVal = asBool(CondVal, CondVT, "whilecond");

    Builder.CreateCondBr(CondBoolVal, LoopBB, AfterBB);
    
    Builder.SetInsertPoint(LoopBB);

    InScopeNames.push_back(std::unordered_map<std::string, llvm::AllocaInst*>());
    
    if (!LoopSection){
        std::cerr << "LoopSection is null" << std::endl;
    }
    //std::cout << "WhileNode part 2.5" << std::endl;
    llvm::Value *bval = LoopSection->codegen();
    //std::cout << "WhileNode part 3" << std::endl;
    
    InScopeNames.pop_back(); // Exit Scope
    Builder.CreateBr(CondBB);

    
    Builder.SetInsertPoint(AfterBB);

    return nullptr;
    }




};

class IfNode : public StmtNode {
  TOKEN Tok;
  std::unique_ptr<ExprNode> Cond;
  std::unique_ptr<BlockNode> TrueSection;
  std::unique_ptr<BlockNode> ElseSection;

  public: IfNode(TOKEN Tok, std::unique_ptr<ExprNode> Cond, std::unique_ptr<BlockNode> TrueSection, 
          std::unique_ptr<BlockNode> ElseSection)
          : Tok(Tok), Cond(std::move(Cond)), TrueSection(std::move(TrueSection)),
          ElseSection(std::move(ElseSection)) {
          }

    virtual std::string to_string(const std::string position="", bool atTail=false) const override {
    std::string temp = setLevel(position, atTail) + "If Stmt\n";
    std::string nextPos = setPosition(position, atTail);
    
    if (Cond) {
        temp += setLevel(nextPos, false) + "Condition\n";
        std::string condPos = setPosition(nextPos, false);
        temp += Cond->to_string(condPos, true);
    }
    
    if (TrueSection) {
        bool thenIsLast = (ElseSection == nullptr);
        temp += setLevel(nextPos, thenIsLast) + "Then\n";
        std::string thenPos = setPosition(nextPos, thenIsLast);
        temp += TrueSection->to_string(thenPos, true);
    }
    
    if (ElseSection) {
        temp += setLevel(nextPos, true) + "Else\n";
        //std::string elsePos = setPosition(nextPos, true);
        temp += ElseSection->to_string(nextPos, atTail);
    }
    
    return temp;
}

    virtual Value *codegen() override { 
        bool returnIf;
        bool returnElse;

        //std::cout << "IfNode" << std::endl;

        llvm::Function *TheFunction = Builder.GetInsertBlock()->getParent();
        //std::cout << "cond gen" << std::endl;
        llvm::Value *CondVal = Cond->codegen();

        //std::cout << "IfNode part 2" << std::endl;
        if(!CondVal){
            return nullptr;
        }
        llvm::Type *CondType = CondVal -> getType();

        CondVal = asBool(CondVal, CondType, "ifcond");

        llvm::BasicBlock *IfBB = llvm::BasicBlock::Create(Context, "if", TheFunction);
        llvm::BasicBlock *ElseBB = ElseSection ? llvm::BasicBlock::Create(Context, "else", TheFunction) : nullptr;
        llvm::BasicBlock *AfterBB = llvm::BasicBlock::Create(Context, "ifcont", TheFunction);

        if (ElseSection != nullptr){
            Builder.CreateCondBr(CondVal, IfBB, ElseBB);
        } else {
            Builder.CreateCondBr(CondVal, IfBB, AfterBB);
        }
        
        InScopeNames.push_back(std::unordered_map<std::string, llvm::AllocaInst*>());
        Builder.SetInsertPoint(IfBB);

        returnIf = TrueSection->codegen() != nullptr;
        Builder.CreateBr(AfterBB);
        IfBB = Builder.GetInsertBlock();

        InScopeNames.pop_back();

        if(ElseBB){ // I.e. False
            InScopeNames.push_back(std::unordered_map<std::string, llvm::AllocaInst*>());
            Builder.SetInsertPoint(ElseBB);
            // Come to 
            
            returnElse = (ElseSection->codegen()) != nullptr;

            ElseBB = Builder.GetInsertBlock();

            //Create a branch to the after block
            Builder.CreateBr(AfterBB);

            InScopeNames.pop_back();
        }

        //TheFunction->getBasicBlockList().push_back(AfterBB);
        Builder.SetInsertPoint(AfterBB);

        if(returnIf && returnElse && ElseSection){
            alwaysReturn = true;
        }else{
            alwaysReturn = false;
        }

        return nullptr; 
    }
};




/* add other AST nodes as nessasary */

//===----------------------------------------------------------------------===//
// Recursive Descent Parser - Function call for each production
//===----------------------------------------------------------------------===//

// Some Forward Declarations

static std::unique_ptr <BlockNode> block();
static std::unique_ptr <StmtNode> stmt();
static std::unique_ptr <BlockNode> else_stmt();
static std::unique_ptr <ExprNode> expr();

/*

-----------    HELPERS     ------------
   

*/
bool inFirstExpr(TOKEN CurTok){
    switch (CurTok.type){
        case NOT:
        case BOOL_LIT:
        case LPAR:
        case IDENT:
        case INT_LIT:
        case FLOAT_LIT:
        case MINUS:
            return true;
        default:
            return false;
    }
}

bool inFollowVFCall(TOKEN Tok){
  //printf("In FollowVFCall with token value: %s\n", Tok.lexeme.c_str());
    switch(Tok.type){
        case SC:
        case RPAR:
        case COMMA:
        case OR:
        case AND:
        case EQ:
        case NE:
        case LE:
        case GE:
        case LT:
        case GT:
        case MINUS:
        case PLUS:
        case ASTERIX:
        case DIV:
        case MOD:
            return true;
        default:
            return false;
    }
}

static TOKEN LookAhead(){
  //printf("Using LookAhead\n");
  TOKEN tmp = CurTok;
  TOKEN next = getNextToken();
  putBackToken(next);

  CurTok = tmp;
  //printf("LookAhead: %s, CurTok : %s, temp: %s\n", next.lexeme.c_str(), CurTok.lexeme.c_str(), tmp.lexeme.c_str());
  return next;
}

const std::unordered_map<TOKEN_TYPE, TYPES> typeSpecMap{
    {INT_TOK, int_T},
    {FLOAT_TOK, float_T},
    {BOOL_TOK, boolean_T},
    {VOID_TOK, void_T}
};

static TYPES type_spec(){
  auto t = static_cast<TOKEN_TYPE>(CurTok.type);
  auto it = typeSpecMap.find(t);

  if(it == typeSpecMap.end()){
    ThrowError(CurTok, "ERROR: Unexpected token -> Expected type");
  } 
  getNextToken();
  return it->second;
  
}


/* Add function calls for each production */



TYPES var_type(){
  TOKEN vt = CurTok;
  getNextToken();
  switch(vt.type){
    case INT_TOK:
      return int_T;
    case FLOAT_TOK:
      return float_T;
    case BOOL_TOK:
      return boolean_T;
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected one of 'bool', 'int', 'float' for variable declaration"));
  }
  exit(1); // Should never reach here, so something has gone wrong
}

/**
  * 
  * End of types
  * 
  * Beginning of Params
  * 
*/

static void ArgpList(std::vector<std::unique_ptr<ExprNode>> &args){

  //printf("In argplist with token value: %s\n", CurTok.lexeme.c_str());

  if (CurTok.type == COMMA){
    getNextToken();
    //printf("Into expr from arglist with token value: %s\n", CurTok.lexeme.c_str());
    auto singarg = expr();
    args.push_back(std::move(singarg));
    ArgpList(args);

  } else if (CurTok.type == RPAR){
    //getNextToken();
    return;
  }
  else{
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ',' or ')' after arg"));
  }

};

static std::vector<std::unique_ptr <ExprNode>> ArgList(){
  std::vector<std::unique_ptr<ExprNode>> args;
  //printf("Into arglist with token value: %s\n", CurTok.lexeme.c_str());
  
  if(CurTok.type ==RPAR){
    return args;
  }

  if (inFirstExpr(CurTok)){
    auto aexpr = expr();
    //printf("%s\n", aexpr->to_string().c_str());
    args.push_back(std::move(aexpr));
    //printf("Into argplist with token value: %s\n", CurTok.lexeme.c_str());
    if(CurTok.type==RPAR){
      return args;
    }
    ArgpList(args);
  }  else{
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected expression or ')' after arg"));
  }
  return args; 
};

static std::unique_ptr<ParamNode> param(){
  TYPES type;
  TOKEN id;

  switch(CurTok.type){
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:
      type = var_type();
      id = CurTok;
      getNextToken();
      //std::cout << "Param finished: " << id.lexeme + " " << CurTok.lexeme << std::endl;
      break;
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected 'int', 'float' or 'boolean' type for a parameter declaration"));
  }
  return std::make_unique<ParamNode>(std::move(type), std::move(id));
}  
static void param_listp(std::vector<std::unique_ptr<ParamNode>> &param_list){
  switch(CurTok.type){
    case COMMA:{
      getNextToken();
      auto Param = param();
      param_list.push_back(std::move(Param));
      param_listp(param_list);
      break;
    }
    case RPAR:
      return;
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ',' or ')' after parameter"));
  }
}  

static std::vector<std::unique_ptr<ParamNode>> param_list(){
  std::vector<std::unique_ptr<ParamNode>> param_list;

  switch(CurTok.type){
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:{
      auto Param = param(); 
      param_list.push_back(std::move(Param));
      param_listp(param_list);
      break;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected 'int', 'float' or 'boolean' type for a parameter list"));
  }

  return param_list;
}
// ::= param_list | "void" | epsilon
static std::vector<std::unique_ptr<ParamNode>> params(){
  std::vector<std::unique_ptr<ParamNode>> params;

  //std::cout << "In params with token value: " << CurTok.type << std::endl;
  //std::cout << CurTok.lexeme << std::endl;
  switch(CurTok.type){
    case RPAR:
      return params;
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:{
      //std::cout<< "In params with type token + "  <<CurTok.lexeme <<std::endl;
      params = param_list();
      break;
      }
    case VOID_TOK:{
      //std::cout<< "In params with VOID_TOK" << std::endl;
      getNextToken();
      break;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected type or ')' after '('"));
  }

  return params;
}

/**
 * Extern functions
 */
// extern = EXTERN type_spec IDENT LPAR params RPAR SC
static:: std::unique_ptr<FuncDecNode> extern_sing(){
    TOKEN id; 
  
  if (CurTok.type != EXTERN){
    ThrowError(CurTok, std::string("ERROR: IN EXTERN \n Unexpected Token -> Expected extern"));
  }

  //printf("Consuming extern\n");
  getNextToken();

  TYPES TypeIden = type_spec();


  if (CurTok.type != IDENT){
    ThrowError(CurTok, std::string("ERROR: IN EXTERN \n Unexpected token -> Expected IDENT after var_type"));
  }
  id = CurTok;
  getNextToken();

  if (CurTok.type != LPAR){
    ThrowError(CurTok, std::string("ERROR: IN EXTERN \n Unexpected token -> Expected '(' after IDENT"));
  }
  getNextToken(); // consume left bracket
  
  //std::cout << "Consumed Left Bracket Before going into extern params" << std::endl;
  
  auto Params = params();

  //std::cout << Params.size() << std::endl;

  if (CurTok.type != RPAR){
    ThrowError(CurTok, std::string("ERROR: IN EXTERN \n Unexpected token -> Expected ') after parameters"));
  }
  getNextToken();
  
  if (CurTok.type != SC){
    ThrowError(CurTok, std::string("ERROR: IN EXTERN \n Unexpected token -> Expected ';' after function declaration"));
  }
  getNextToken();
  //std::cout << "EXTERN FIN " << CurTok.lexeme << std::endl; 
  return std::make_unique<FuncDecNode>(std::move(TypeIden), std::move(id), std::move(Params), nullptr);
}

// extern'_list = extern extern'_list | epsilon
static void externp_list(std::vector<std::unique_ptr<FuncDecNode>> &externs){
  switch(CurTok.type){
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:
    case VOID_TOK:
      //std::cout << "Encountered TOK" << CurTok.lexeme << std::endl;
      return;
    case EXTERN:{
      auto s = extern_sing();
      externs.push_back(std::move(s));
      externp_list(externs);
      break;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected extern or type"));
  }
}

// extern_list = extern extern'_list
static std::vector<std::unique_ptr<FuncDecNode>> extern_list(){
  std::vector<std::unique_ptr<FuncDecNode>> externs;

  if (CurTok.type == EXTERN) {
    auto s = extern_sing();
    externs.push_back(std::move(s));
    //std::cout << "Pushing back extern" << std::endl;
    externp_list(externs);

  }
  else {
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected extern"));
  }
  //std::cout<< "Finished Externs, on token: " << CurTok.lexeme << std::endl;
  return externs;
}

/**
 * End of Externs
 * 
 * Beginning of delc functions
 */

// It begins

static std::unique_ptr<ExprNode> identCall(){
  TOKEN id = CurTok;
  getNextToken();
  
  if(CurTok.type == LPAR){  
    
    //LPAR 
    getNextToken();
    //std::cout << "In identCall with token value: " << CurTok.lexeme << std::endl;

    auto args = ArgList();
    
    //printf("%s\n", CurTok.lexeme.c_str());
    if(CurTok.type != RPAR){
      ThrowError(CurTok, "ERROR: Unexpected token -> Expected ')' after args in function call");
    }
    getNextToken();

    return std::make_unique<FuncCallNode>(std::move(id), std::move(args));
 
  } else if (inFollowVFCall(CurTok)){ // if in FOLLOW
      return std::make_unique<VarValueNode>(std::move(id));
    // Variable Call -> want value of variable
  } else if(CurTok.type == RPAR){
    return std::make_unique<VarValueNode>(std::move(id));
  } 
  else{
    ThrowError(CurTok, "ERROR: Unexpected token -> Expected '(' or epsilon after IDENT");
  }

  //std::cout << "Empty identCall\n";

  return nullptr;
}

static std::unique_ptr<ExprNode> litCallExpr(){
  std::unique_ptr<ExprNode> expression;
  TOKEN lit = CurTok;
  //std::string errMsg;

  switch (CurTok.type){ 
    case INT_LIT:{
      getNextToken();
      expression = std::make_unique<IntNode>(std::move(lit));
      //errMsg = "ERROR: Unexpected token -> Expected INT_LIT";
      break;
    }
    case FLOAT_LIT:{
      getNextToken();
      expression = std::make_unique<FloatLitNode>(std::move(lit));
      //errMsg = "ERROR: Unexpected token -> Expected FLOAT_LIT";
      break;
    }
    case BOOL_LIT:{
      getNextToken();
      expression = std::make_unique<BoolLitNode>(std::move(lit));  
      //errMsg = "ERROR: Unexpected token -> Expected BOOL_LIT";
      break;
    }
    case IDENT:{
      expression = identCall();
      break;
    }
    case LPAR:{
      getNextToken();
      
      expression = expr();

      if(CurTok.type != RPAR){
        ThrowError(CurTok, "ERROR: Unexpected token -> Expected ')' after bidmas expression");
      }
      //printf("Consuming RPAR\n");
      getNextToken();
      break;  
    }
    default:
      ThrowError(CurTok, "ERROR: Unexpected token -> Expected INT_LIT, FLOAT_LIT, BOOL_LIT");
  }

  return expression; 
};


static std::unique_ptr<ExprNode> neg(){
  std::unique_ptr<ExprNode> expr;
  TOKEN operate;
  //printf("In neg with token value: %s\n", CurTok.lexeme.c_str());

  switch(CurTok.type){
    case MINUS:{
      operate = CurTok;
      getNextToken();
      expr = neg();
      return std::make_unique<UnaryNode>(std::move(expr), std::move(operate));
    }
    case NOT:{
      operate = CurTok;
      getNextToken();
      expr = neg();
      return std::make_unique<UnaryNode>(std::move(expr), std::move(operate));
    }
    case LPAR:
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case IDENT:
      return litCallExpr();
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '!', '-', '(', INT_LIT, FLOAT_LIT, BOOL_LIT or IDENT after expression"));
  }
  return nullptr;
};

static std::unique_ptr<ExprNode> multdivp(std::unique_ptr<ExprNode> lhs){
  std::unique_ptr<ExprNode> lp;
  std::unique_ptr<ExprNode> rp;
  std::unique_ptr<ExprNode> expr;
  TOKEN operate = CurTok;

  switch(CurTok.type){
    case ASTERIX:
    case DIV:
    case MOD:{
      //printf("IN MULTDIVP\n CurTok: %s\n", CurTok.lexeme.c_str());
      getNextToken();
      lp = neg();
      
      rp = multdivp(std::move(lhs));
      expr = std::make_unique<BinaryExprNode>(std::move(lp), std::move(rp), std::move(operate));
      return expr;
    }
    case PLUS:
    case MINUS:
    case LT:
    case GT:
    case GE:
    case LE:
    case EQ:
    case NE:
    case AND:
    case OR:
    case COMMA:
    case RPAR:
    case SC:{
      //printf("Back from multidivp as not adm operator\n");
      expr = std::move(lhs);
      return expr;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token in multdivp -> Expected  '*', '/', '%', '+', '-', '<', '>', '<=', '>=', '==', '!=', '&&', '||', ',', ')' or ';' after expression"));
  }
  return nullptr;
};

static std::unique_ptr<ExprNode> multdiv(){
  std::unique_ptr<ExprNode> lhs = neg();

  while(CurTok.type == ASTERIX || CurTok.type == DIV || CurTok.type == MOD){
      TOKEN operate = CurTok;
      getNextToken();
      auto rhs = neg();
      lhs = std::make_unique<BinaryExprNode>(std::move(lhs), std::move(rhs), std::move(operate));
  }
  return multdivp(std::move(lhs));
};

// It begins
static std::unique_ptr<ExprNode> addexprp(std::unique_ptr<ExprNode> lhs){
  std::unique_ptr<ExprNode> lp;
  std::unique_ptr<ExprNode> rp;
  std::unique_ptr<ExprNode> expr;
  TOKEN operate = CurTok;

  switch(CurTok.type){
    case PLUS: 
    case MINUS:{
      //printf("Consuming Arithmetic Token \n");
      getNextToken();
      lp = multdiv();
      rp = addexprp(std::move(lhs));
      expr = std::make_unique<BinaryExprNode>(std::move(lp), std::move(rp), std::move(operate));
      return expr;
    }
    case LT:
    case GT:
    case GE:
    case LE:
    case COMMA:
    case RPAR:
    case SC:
    case EQ:
    case NE:
    case AND:
    case OR:{
      expr = std::move(lhs);
      return expr;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token, made it to arithmetic processing -> Expected '+', '-', '<', '>', '<=', '>=', '==', '!=', '&&', '||', ',', ')' or ';' after expression"));
  }
  return nullptr;

};

static std::unique_ptr<ExprNode> addexpr(){
  std::unique_ptr<ExprNode> lhs = multdiv();

  while (CurTok.type == PLUS || CurTok.type == MINUS){
    TOKEN operate = CurTok;
    //printf("Consuming PLUS or MINUS\n");
    getNextToken();
    auto rhs = multdiv();
    lhs = std::make_unique<BinaryExprNode>(std::move(lhs), std::move(rhs), std::move(operate));
  };
  return addexprp(std::move(lhs));
};


static std::unique_ptr<ExprNode> relexprp(std::unique_ptr<ExprNode> lhs){
  std::unique_ptr<ExprNode> lp;
  std::unique_ptr<ExprNode> rhs;
  std::unique_ptr<ExprNode> expr;
  TOKEN operate = CurTok;

  switch(CurTok.type){
    case LT:
    case GT:
    case LE:
    case GE:{
      getNextToken();
      lp = addexpr();
      rhs = relexprp(std::move(lhs));
      expr = std::make_unique<BinaryExprNode>(std::move(lp), std::move(rhs), std::move(operate));
      return expr;
    }
    case EQ:
    case NE:
    case AND:
    case OR:
    case COMMA:
    case RPAR:
    case SC:{
      expr = std::move(lhs);
      return expr;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '<', '>', '<=', '>=', '==', '!=', '&&', '||', ',', ')' or ';' after expression"));
  }
  return nullptr;
};

static std::unique_ptr<ExprNode> relexpr(){
  std::unique_ptr<ExprNode> lhs = addexpr();

  while (CurTok.type == LT || CurTok.type == GT || CurTok.type == LE || CurTok.type == GE){
    TOKEN operate = CurTok;
    getNextToken();
    auto rhs = addexpr();
    lhs = std::make_unique<BinaryExprNode>(std::move(lhs), std::move(rhs), std::move(operate));
  }

  return relexprp(std::move(lhs));

};


//I've missed so many semicolons that its becoming recommended to miss them


static std::unique_ptr<ExprNode> logeqp(std::unique_ptr<ExprNode> lhs){
  std::unique_ptr<ExprNode> lp;
  std::unique_ptr<ExprNode> rhs;
  std::unique_ptr<ExprNode> expr;
  TOKEN operate = CurTok;

  switch(CurTok.type){
    case EQ:
    case NE:{
      getNextToken();
      lp = relexpr();
      rhs = logeqp(std::move(lhs));
      expr = std::make_unique<BinaryExprNode>(std::move(lp),std::move(rhs),std::move(operate));
      return expr;
    }
    case AND:
    case OR:
    case COMMA:
    case RPAR:
    case SC:{
      expr = std::move(lhs);
      return expr;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '==', '!=', '&&', '||', ',', ')' or ';' after expression"));
  }
  return nullptr;
}

static std::unique_ptr<ExprNode> logeq(){

  std::unique_ptr<ExprNode> lhs = relexpr();
  
  while (CurTok.type == EQ || CurTok.type == NE){
    TOKEN operate = CurTok;
    getNextToken();
    //printf("Consuming EQ or NE token\n new = %s \n", CurTok.lexeme.c_str());
    
    auto rhs = relexpr();

    lhs = std::make_unique<BinaryExprNode>(std::move(lhs), std::move(rhs), std::move(operate));
  }
  return logeqp(std::move(lhs));
};



static std::unique_ptr<ExprNode> logandp(std::unique_ptr<ExprNode> lhs){
  std::unique_ptr<ExprNode> lp;
  std::unique_ptr<ExprNode> rhs;
  std::unique_ptr<ExprNode> expr;
  TOKEN op;

  switch(CurTok.type){
    case AND:
      op = CurTok;
      getNextToken();
      lp = logeq();
      rhs = logandp(std::move(lhs));
      expr = std::make_unique<BinaryExprNode>(std::move(lp), std::move(rhs), std::move(op));
      return expr;
    case OR:
    case COMMA:
    case RPAR:
    case SC:
      expr = std::move(lhs);
      return expr;
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '&&', ',', ')' or ';' after expression"));
  }
  return nullptr;
};

static std::unique_ptr<ExprNode> logand(){
  std::unique_ptr<ExprNode> lhs = logeq();
  
  while (CurTok.type == AND){ // if chaining expressions are logical and operations
    TOKEN operate = CurTok;
    getNextToken();
    auto rhs = logeq();
    lhs = std::make_unique<BinaryExprNode>(std::move(lhs), std::move(rhs), std::move(operate));
  }

  return logandp(std::move(lhs));

}

static std::unique_ptr<ExprNode> logorp(std::unique_ptr<ExprNode> lhs){
  std::unique_ptr<ExprNode> lp;
  std::unique_ptr<ExprNode> rhs;
  std::unique_ptr<ExprNode> expr;
  TOKEN op;
  //printf("logorp, %s\n", CurTok.lexeme.c_str());
  switch(CurTok.type){
    case OR:
      op = CurTok;
      getNextToken();
      
      lp = logand();
      rhs = logorp(std::move(lhs));
      expr = std::make_unique<BinaryExprNode>(std::move(lp), std::move(rhs), std::move(op));
      return expr;
    case COMMA: 
    case RPAR: 
    case SC:
      expr = std::move(lhs);
      return expr;
    default: 
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '||', ',', ')' or ';' after expression"));
  }
  return nullptr;
}


static std::unique_ptr<ExprNode> logor(){
  
  std::unique_ptr<ExprNode> lhs = logand();
  TOKEN op = CurTok;

  if (CurTok.type == OR){
    getNextToken();
    std::unique_ptr<ExprNode> rhs = logor();
    return std::make_unique<BinaryExprNode>(std::move(lhs), std::move(rhs), std::move(op));
  }

  auto eval = logorp(std::move(lhs));
  return std::move(eval);
}


static std::unique_ptr<ExprNode> expr(){
  std::unique_ptr<ExprNode> base;
  //printf("%s \n", CurTok.lexeme.c_str());
  //printf("In Expression Def\n");
  //print(CurTok, "EXPR \n");

  if(CurTok.type==SC){
    getNextToken();
    return base;
  }

  switch(CurTok.type){
    case LPAR:
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case NOT:
    case MINUS:{
      base = logor();
      break;
    }
    case IDENT:{
      //std::cout << CurTok.lexeme << std::endl;
      TOKEN next = LookAhead();
      //std::cout << CurTok.lexeme << std::endl;
      
      if (next.type == ASSIGN){
        //printf("Assignment\n");
        TOKEN id = CurTok;
        if(id.type != IDENT){
          ThrowError(CurTok, std::string("ERROR: IN EXPR DEF \n Unexpected token -> Expected IDENT"));
        }
        //printf("Consuming IDENT\n");
        getNextToken();

        // no need for check we know its ASSIGN
        //printf("Consuming ASSIGN\n %s\n", CurTok.lexeme.c_str());
        getNextToken();
        //printf("Going into expr again, %s\n", CurTok.lexeme.c_str());
        auto vexpr = expr();
        //printf("Out of expr \n");
        base = std::make_unique<VarCallNode>(std::move(id), std::move(vexpr));
      } else{

        //printf("Into bidmas exprs \n");
        base = logor();
        return base;
        //printf("EXPR FINISHED CURTOKEN: %s\n", CurTok.lexeme.c_str());
      }
      break;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: IN EXPR DEF \n Unexpected token -> Expected expression or ';'"));
  }
  return base;
}

static std::unique_ptr<ExprNode> expr_stmt(){
  std::unique_ptr<ExprNode> exprp;
  switch (CurTok.type){
    case SC:
      getNextToken();
      break;
    case LPAR:
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case NOT:
    case MINUS:
    case IDENT:
      exprp = expr();
      if(CurTok.type != SC){
        ThrowError(CurTok, std::string("ERROR: AFTER EXPRESSION DEF \n Unexpected token -> Expected ';' \n"));
      }
      getNextToken();
      break;
  default:
    ThrowError(CurTok, std::string("ERROR: start of expression definition \n Unexpected token -> Expected expression or ';'"));
  }
  return exprp;
}

static std::unique_ptr<ReturnNode> returnFinal(){
  TOKEN RTok = CurTok;
  std::unique_ptr<ExprNode> retExpr;
  switch(CurTok.type){
    case SC:{
      RTok = CurTok;
      getNextToken();
      return std::make_unique<ReturnNode>(std::move(retExpr), std::move(RTok));
    } 
    case LPAR:
    case IDENT:
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case NOT:
    case MINUS:{
      //std::cout << CurTok.lexeme  << std::endl;
      retExpr = expr();
      return std::make_unique<ReturnNode>(std::move(retExpr), std::move(RTok));
    }
    default:
      ThrowError(CurTok, std::string("ERROR: IN RETURN \n Unexpected token -> Expected expression or ';' after 'return'"));
  }
  return nullptr;
}

static std::unique_ptr<ReturnNode> return_stmt(){
  std::unique_ptr<ReturnNode> ret;
  
  if(CurTok.type != RETURN){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected 'return' to start return statement"));
  }
  getNextToken();
  ret = returnFinal();
  return ret;
}

static std::unique_ptr<WhileNode> while_stmt(){
  std::unique_ptr<ExprNode> cond;
  std::unique_ptr<StmtNode> loopSection;
  TOKEN id = CurTok;

  if (CurTok.type != WHILE){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected 'while' to start while loop"));
  }
  getNextToken();
  if(CurTok.type != LPAR){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '(' after 'while'"));
  }
  getNextToken();

  cond = expr();
  //std::cout << "while cond="<< cond->to_string() << std::endl;

  if (CurTok.type != RPAR){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ')' after condition"));
  }
  getNextToken();

  loopSection = stmt();


  return std::make_unique<WhileNode>(std::move(id), std::move(cond), std::move(loopSection));

}

static std::unique_ptr<IfNode> if_stmt(){
  std::unique_ptr<ExprNode> cond;
  std::unique_ptr<BlockNode> trueSection;
  std::unique_ptr<BlockNode> elseSection;

  TOKEN id = CurTok;
  //printf("In if_stmt, CurTok=%s \n", CurTok.lexeme.c_str());

  if(CurTok.type != IF){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected 'if' to start if statement"));
  }
  getNextToken();

  if(CurTok.type != LPAR){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '(' after 'if'"));
  }
  getNextToken();

  //printf("Cond first tok = %s\n", CurTok.lexeme.c_str());
  
  //std::cout << "In if_stmt with token value: " << CurTok.lexeme << std::endl;
  cond = expr();
  if(cond == nullptr){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected expression after '('"));
  }
  //std::cout << cond->to_string() << std::endl;

  //std::cout << "If stmt expr finished" << std::endl;

  if(CurTok.type != RPAR){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ')' after condition"));
  }
  getNextToken();

  trueSection = block();
  elseSection = else_stmt();
  //std::cout << "returning if node\n";
  //std::cout << cond->to_string() << std::endl;
  //std::cout << std::make_unique<IfNode>(std::move(id), std::move(cond), std::move(trueSection), std::move(elseSection))->to_string() << std::endl;
  
  //std::cout << "returning if node\n";

  return std::make_unique<IfNode>(std::move(id), std::move(cond), std::move(trueSection), std::move(elseSection));

}

static std::unique_ptr<BlockNode> else_stmt(){
  //std::unique_ptr<BlockNode> elseSection;

  switch(CurTok.type){
    case ELSE:{
      getNextToken();
      //Only create the Block if needed 
      std::unique_ptr<BlockNode> elseSection = block();
      return elseSection;
    } 
    case IDENT:
    case IF:
    case WHILE:
    case RETURN:
    case LPAR:
    case LBRA:
    case RBRA:
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case MINUS:
    case NOT:
    case SC:
      return nullptr;
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected 'else' or statement e.g. {IDENT, MINUS, WHILE, IF, INT_LIT}"));
  }
  return nullptr;
}

static std::unique_ptr<StmtNode> stmt(){
  std::unique_ptr<StmtNode> stmt;

  //std::cout << "STMT PICK" << CurTok.lexeme << std::endl;

  switch(CurTok.type){
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case MINUS: 
    case NOT:
    case LPAR:
    case IDENT:
    case SC:{
      //printf("Expression\n");
      stmt = expr_stmt();
      //printf("stmt=expr()");
      break;
    }
    case RETURN:{
      stmt = return_stmt();
      break;
    } 
    case IF:{
      stmt = if_stmt();
      break; 
    }
    case WHILE:{
      stmt = while_stmt();
      break;
    } 
    case LBRA:{
      stmt = block();
      break;
    } 
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected statement e.g. {IDENT, MINUS, WHILE, IF, INT_LIT}"));
  }
  return stmt;
};

static void stmt_list(std::vector<std::unique_ptr <StmtNode>> &slist){
  switch(CurTok.type){
    case IDENT:
    case IF:
    case WHILE:
    case RETURN:
    case LPAR:
    case LBRA:
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case MINUS:
    case NOT:
    case SC:{
      auto sect = stmt();
      
      if(sect != nullptr){
        slist.push_back(std::move(sect));
      }
      stmt_list(slist);
      break;
    }    
    case RBRA:
      break; // was return; before 
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected statement e.g. {IDENT, MINUS, WHILE, IF, INT_LIT}"));
  }
};

static std::unique_ptr<VarDecNode> local_decl(){
  TYPES type;
  TOKEN id;
  //printf("Local Decl\n");
  switch(CurTok.type){
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:{
      type = var_type();
      
      id = CurTok;
      if (id.type != IDENT){
        ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected IDENT after type"));
      }
      getNextToken();

      if (CurTok.type != SC){
        ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ';' after variable declaration"));
      }
      //printf("%s\n", CurTok.lexeme.c_str());
      getNextToken();
      break;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected 'int', 'float' or 'boolean' type for a variable declaration"));
  }
  //printf("Returning VarDec\n %s\n %s\n", std::make_unique<VarDecNode>(std::move(type), std::move(id))->to_string().c_str(), id.lexeme.c_str());
  return std::make_unique<VarDecNode>(std::move(type), std::move(id));
};


static void local_decls(std::vector<std::unique_ptr<VarDecNode>> &localVDecs){
  
  switch(CurTok.type){
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:{
      auto local = local_decl();
      localVDecs.push_back(std::move(local));
      local_decls(localVDecs);
      break;
    }
    case IDENT:
    case IF:
    case WHILE:
    case RETURN:
    case LPAR:
    case LBRA:
    case RBRA:
    case BOOL_LIT:
    case INT_LIT:
    case FLOAT_LIT:
    case MINUS:
    case NOT:
    case SC:
      return;
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected declaration, or statement e.g. {IDENT, MINUS, WHILE, IF, INT_LIT}"));
      exit(1);
  };
}


static std::unique_ptr<BlockNode> block(){
  if (CurTok.type != LBRA){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '{' to start block for new scope"));
  }
  getNextToken();
  
  // ^ avoids declaring vectors if an error occurs

  std::vector<std::unique_ptr<VarDecNode>> localVDecs;
  local_decls(localVDecs);
  //printf("Out of Local_decls \n");



  std::vector<std::unique_ptr<StmtNode>> localStmts;
  stmt_list(localStmts);
  
  

  if (CurTok.type != RBRA){
    ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '}' to end scope block"));
  }
  getNextToken();

  return std::make_unique<BlockNode>(std::move(localVDecs), std::move(localStmts));
}



static void declp(std::unique_ptr<FuncDecNode> &fdec, std::unique_ptr<VarDecNode> &vdec, TYPES type, TOKEN id){
  switch(CurTok.type){
    case LPAR:{
      getNextToken();

      //std::cout << "Consuming LPAR Before param_list" << std::endl;

      auto Params = params();
      if (CurTok.type != RPAR){
        ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ')' after parameters"));
      }
      getNextToken();
      auto defblock = block();
      //printf("Making FuncDec\n");
      fdec = std::make_unique<FuncDecNode>(std::move(type), std::move(id), std::move(Params), std::move(defblock));
      break;
    } 
    case SC:
      getNextToken();
      vdec = std::make_unique<VarDecNode>(std::move(type), std::move(id));
      break;
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ';' or parameters for a function after type"));
  }
}

static std::unique_ptr<DeclNode> decl(){
  // original ::= fdec | vdec
  /* now, 
        decl -> typespec IDENT decl'
        decl' -> "(" params ")" block 
              | SC
  */
  std::unique_ptr<FuncDecNode> fdec;
  std::unique_ptr<VarDecNode> vdec;
  //printf("Decl\n");
  switch(CurTok.type){
    case VOID_TOK: {//Only void for functions
      //std::cout<< "In decl with VOID_TOK" << std::endl;
      getNextToken();
      if(CurTok.type != IDENT){
        ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected IDENT after void"));
      }
      //TYPES type = void_T;
      TOKEN id = CurTok;
      //std::cout << "Function name: " << id.lexeme << std::endl;
      getNextToken();
      if(CurTok.type != LPAR){
        ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected '(' after IDENT"));
      }
      getNextToken();
      auto Params = params();
      if(CurTok.type != RPAR){
        ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected ')' after parameters"));
      }
      getNextToken();
      auto dblock = block();
      //std::cout << id.lexeme << " Function Parsed\n";
      fdec = std::make_unique<FuncDecNode>(void_T, std::move(id), std::move(Params), std::move(dblock));
      // using types enum instead of token type
      break;
    } // other cases could be either
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:{
      auto type = var_type();
      //getNextToken is within var_type

      if (CurTok.type != IDENT){
        ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected IDENT after type"));
      }
      TOKEN id = CurTok;
      getNextToken();
      declp(fdec, vdec, type, id);
      break;    
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected variable or function type declaration"));
  }
  return std::make_unique<DeclNode>(std::move(fdec), std::move(vdec));

};

static void declp_list(std::vector<std::unique_ptr<DeclNode>> &declarations){
  switch(CurTok.type){
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:
    case VOID_TOK:{
      auto single = decl();
      //std::cout << "Pushing decl\n";
      declarations.push_back(std::move(single));
      declp_list(declarations);
      break;
    }
    case EOF_TOK:{
      //printf("End of File Reached\n");
      return;
    }
    default:
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected type"));
  }
}

static std::vector<std::unique_ptr<DeclNode>> decl_list(){
  std::vector<std::unique_ptr<DeclNode>> declarations;

  switch(CurTok.type){
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:
    case VOID_TOK:{
      auto single = decl();
      declarations.push_back(std::move(single));
      declp_list(declarations);
      break;
    } 
    default:
      ThrowError(CurTok, std::string("ERROR: In decl_list \nUnexpected token -> Expected type"));
  }
  //std::cout << "Returning decls \n";
  return declarations;
}


static std::unique_ptr<ProgramNode> program(){
  std::vector<std::unique_ptr<FuncDecNode>> externs;
  std::vector<std::unique_ptr<DeclNode>> decls;
 
  switch(CurTok.type){
    case EXTERN:
      externs = extern_list();
      decls = decl_list();
      break;
    case INT_TOK:
    case FLOAT_TOK:
    case BOOL_TOK:
    case VOID_TOK:
      decls = decl_list();
      break;
    default:
      //printf("%s , \n %d \n\n",CurTok.lexeme.c_str(), CurTok.type);
      ThrowError(CurTok, std::string("ERROR: Unexpected token -> Expected type or 'extern'"));
  }
  return std::make_unique<ProgramNode>(std::move(externs), std::move(decls));
  ///DO NOT DELETE

  
 return nullptr;
}

/*
 * End of Params
 * 
 * Beginning of Scoped Sections
 */


/*
 * End of Scope Sections 
 * 
 * Beginning of Control Flow Function Functions
 * 
 */

// 






/**
 * 
 * End of Control Flow Functions
 * 
 * Beginning of Base Expr + Expr Functions
 * 
 */




/**
 *  ----------------- End of Base Expr + Expr Functions -----------------
 * 
 * ------------------------ Beginning of Args ----------------------------
 */

/*
bool inFirstExpr(){
    switch (CurTok.type){
        case NOT:
        case BOOL_LIT:
        case LPAR:
        case IDENT:
        case INT_LIT:
        case FLOAT_LIT:
        case MINUS:
            return true;
        default:
            return false;
    }
} */

/*
 *
 * Helper
 *   
 */



/*
 liter ::= INT_LIT | FLOAT_LIT | BOOL_LIT
        | IDENT
        | IDENT "(" args ")"
        | "(" expr ")"
*/

// program ::= extern_list decl_list

/**
 * 
 * Parser Function 
 * 
 */


;

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//
/*/
static LLVMContext Context;
static IRBuilder<> Builder(TheContext);
static std::unique_ptr<Module> TheModule;  */

//===----------------------------------------------------------------------===//
// AST Printer
//===----------------------------------------------------------------------===//

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ASTnode &ast) {
  os << ast.to_string();
  return os;
  //.c_str() ?
  //llvm::StringRef(...)?
  // 
} 

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

static std::unique_ptr<ProgramNode> rootNode;

static void parser() {
  std::cout<<"Starting Parser" << std::endl;
  getNextToken();

  
  rootNode = program(); 
  if (CurTok.type != EOF_TOK){
      std::cerr << "ERROR: Expected EOF token \n\n";
    } 
  //printf("finaltok = %s\n", CurTok.lexeme.c_str());
  
  if (rootNode != nullptr){
    try{
      //printf("AST:\n %s\n", rootNode->to_string().c_str());
      std::cout << "AST:\n" << (rootNode->to_string() )<< std::endl;
      std::cout << "Done Printing AST" << "\n";
    } catch (const std::exception &e){
      std::cerr << "ERROR: " << e.what() << std::endl;
    }   
    std::cout << "Parsing Finished Without Errors \n\n";
  } else {
    std::cerr << "Parsing Finished Unsuccessfully \n\n";
  }
  
}

bool PIR = true; // Print Intermediate Representation
bool PAST = true; // Print Abstract Syntax Tree
bool PALE = true; // Print Alerts 

int main(int argc, char **argv) {
  if (argc == 2) {
    pFile = fopen(argv[1], "r");
    if (pFile == NULL)
      perror("Error opening file");
  } else {
    std::cout << "Usage: ./code InputFile\n";
    return 1;
  }

  // initialize line number and column numbers to zero
  lineNo = 1;
  columnNo = 1;

  // Make the module, which holds all the code.
  TheModule = std::make_unique<Module>("mini-c", Context);

  
  // Run the parser now.
  parser();
  fprintf(stderr, "Parsing Finished\n");

  std::cout<<"Codegen Beginning" << std::endl;
  rootNode->codegen();

  //********************* Start printing final IR **************************
  // Print out all of the generated code into a file called output.ll
  auto Filename = "output.ll";
  std::error_code EC;
  raw_fd_ostream dest(Filename, EC, sys::fs::OF_None);
  bool PIR = true;
  bool PALE = true;

  if (EC) {
    errs() << "Could not open file: " << EC.message();
    return 1;
  }

  if (PIR){
    TheModule->print(errs(), nullptr);
  }

  if (PALE){
    if(!Alerts.empty()){
      for (const auto &alert : Alerts){
        alert.print();
      }
    };
  }

  // TheModule->print(errs(), nullptr); // print IR to terminal
  TheModule->print(dest, nullptr);
  //********************* End printing final IR ****************************

  fclose(pFile); // close the file that contains the code that was parsed
  return 0;
}


