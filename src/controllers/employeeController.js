const Employee = require('../models/Employee')
const mongoose = require('mongoose');
const Career = require('../models/Career')
const calculateAge = require('../public/js/calcAge')


/**
 * GET /
 * employee page
 */


exports.employee = async (req, res) => {

    const messages = await req.flash('info');

    const locals = {
      title: "About - NodeJs",
      description: "Free NodeJS User Management System.",
    }

    let perPage = 12;
    let page = req.query.page || 1;


//페이지에 보여줄 작업자 수
    try {
      const employees = await Employee.aggregate([ { $sort: {updatedAt: -1 } } ])
        .skip(perPage * page - perPage)
        .limit(perPage)
        .exec();
      const count = await Employee.countDocuments({});

      res.render('employee/employee', {
        locals,
        employees,
        current: page,
        pages: Math.ceil(count / perPage),
        messages
      });
    } catch (error) {
      console.log(error);
    }
  }

/**
 * GET /
 * 새로운 노동자 양식
 */

exports.addEmployee = async (req, res) => {
    const locals = {
      title: "Add New Employee",
      description: "Free Nodejs User Management System.",
    }
    res.render('employee/addemployee', locals);
  }

/**
 * POST /
 * 새로운 노동자 생성
 */

exports.postEmployee = async (req, res) => {

    console.log(req.body);

    const newEmployee = new Employee({
        user: req.user._id,
        name: req.body.name,
        sex: req.body.sex,
        local: req.body.local,
        RRN: req.body.RRN,
        phonenumber: req.body.phonenumber,
    });

    try {
        req.body.user = req.user.id;
        await Employee.create(newEmployee);
        await req.flash('info', '새 작업자가 추가되었습니다.')

        res.redirect('/employee');
    } catch (error) {
        console.log(error);
    }
    
  };

/**
 * GET /
 * 노동자 데이터
 *  */

  exports.viewEmployee = async (req, res) => {

    try{
      const employee = await Employee.findOne({ _id: req.params.id })
      
      const locals = {
        title: "View Employee Data",
        description: "Free NodeJs User Management System",
      };
      // const careers = await Career.find({employee: req.params.id}).populate('worksite')
      // console.log(careers)
      res.render('employee/viewemployee', {
        locals,
        employee,
        calculateAge,
      })
    } catch(error) {
      console.log(error);
    }
  };

  /**
 * GET /
 * 노동자 데이터 수정
 *  */

  exports.editEmployee = async (req, res) => {

    try{
      const employee = await Employee.findOne({ _id: req.params.id })

      const locals = {
        title: "Edit Employee Data",
        description: "Free NodeJs User Management System",
      };

      res.render('employee/editemployee', {
        locals,
        employee
      })
    } catch(error) {
      console.log(error);
    }
  };

    /**
 * GET /
 * 노동자 데이터 업데이트
 *  */
    exports.editPostemployee = async (req, res) => {
      try {
        await Employee.findOneAndUpdate(
          { _id: req.params.id },
          { name: req.body.name,
            sex: req.body.sex,
            local: req.body.local,
            RRN: req.body.RRN,
            phonenumber: req.body.phonenumber,
            // details: req.body.details,
            updatedAt: Date.now(),
          }
        ).where({ user: req.user.id });
        res.redirect("/employee");
      } catch (error) {
        console.log(error);
      }
    };
    // exports.editPostemployee = async (req, res) => {

    //   try {

    //     await Employee.findByIdAndUpdate(req.params.id, {
    //       name: req.body.name,
    //       sex: req.body.sex,
    //       local: req.body.local,
    //       RRN: req.body.RRN,
    //       phonenumber: req.body.phonenumber,
    //       // details: req.body.details,
    //       updatedAt: Date.now(),
    //     });
    //     await res.redirect(`/editemployee/${req.params.id}`);

    //     console.log('redirected');
    //   } catch (error) {
    //     console.log(error);
    //   }
    // };

       /**
 * DELETE /
 * 노동자 데이터 삭제
 *  */

       exports.deleteEmployee = async (req, res) => {
        try {
          await Employee.deleteOne({ _id: req.params.id });
          res.redirect("/employee")
        } catch (error) {
          console.log(error);
        }
      };

             /**
 * Get /
 * 노동자 데이터 검색
 *  */

        exports.searchEmployee = async (req, res) => {
          const locals = {
            title: "Search Employee Data",
            description: "Free NodeJs User Management System",
          };

          try {
            let searchTerm = req.body.searchTerm;
            const searchNoSpcialChar = searchTerm.replace(/[^a-zA-Z0-9]/g, "");
            const employees = await Employee.find({
              $or: [
                { name: { $regex: new RegExp(searchNoSpcialChar, "i") }},

            ],
          });

          res.render("employee/searchemployee", {
            employees,
            locals,
          });

          } catch (error) {
            console.log(error);
          }
        };