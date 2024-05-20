/**
 * GET /
 * Homepage 
*/
exports.homepage = async (req, res) => {
  const locals = {
    title: "NodeJs Notes",
    description: "Free NodeJS Notes App.",
  }
  res.render('index', {
    locals,
    layout: '../views/layouts/front-page'
  });
}


/**
 * GET /
 * About 
*/
exports.about = async (req, res) => {
  const locals = {
    title: "About - NodeJs Notes",
    description: "Free NodeJS Notes App.",
  }
  res.render('about', locals);
}

//GET / WORKSITE

exports.worksite = async (req, res) => {
  const locals = {
    title: "About - NodeJs Notes",
    description: "Free NodeJS Notes App.",
  }
  res.render('worksite', locals);
}

//GET EMPLOYEE

exports.employee = async (req, res) => {
  const locals = {
    title: "About - NodeJs Notes",
    description: "Free NodeJS Notes App.",
  }
  res.render('employee', locals);
}