require=function e(t,r,o){function n(s,a){if(!r[s]){if(!t[s]){var c="function"==typeof require&&require;if(!a&&c)return c(s,!0);if(i)return i(s,!0);var u=new Error("Cannot find module '"+s+"'");throw u.code="MODULE_NOT_FOUND",u}var p=r[s]={exports:{}};t[s][0].call(p.exports,(function(e){return n(t[s][1][e]||e)}),p,p.exports,e,t,r,o)}return r[s].exports}for(var i="function"==typeof require&&require,s=0;s<o.length;s++)n(o[s]);return n}({1:[function(e,t,r){t.exports=YUI_config.flickr},{}],2:[function(e,t,r){t.exports=function(){var e=new Date,t="";switch(e.getMonth()+1+"-"+e.getDate()){case"2-1":t="is-chinesenewyears-day";break;case"2-10":t="is-flickr-day";break;case"2-14":t="is-valentines-day";break;case"3-17":t="is-patties-day";break;case"4-22":t="is-earth-day";break;case"6-20":case"6-21":case"6-22":case"6-23":case"6-24":case"6-25":case"6-26":t="is-pride-day";break;case"7-1":t="is-canada-day";break;case"7-4":t="is-usa-day";break;case"7-14":t="is-french-day";break;case"9-19":t="is-pirate-day";break;case"10-31":t="is-halloween-day";break;case"12-18":case"12-19":case"12-20":case"12-21":case"12-22":case"12-23":case"12-26":t="is-hanukkah-day";break;case"12-24":case"12-25":t=Math.random()<.5?"is-xmas-day":"is-hanukkah-day"}return t&&(t=" "+t),t}},{}],3:[function(e,t,r){var o=e("hermes-core/type-validator");t.exports=function(e){return function(t,r,n){var i=t.params[e];if(!i||"string"!=typeof i)return n();o.nsid(i)||(t.params[e]=String(i).toLowerCase()),n()}}},{"hermes-core/type-validator":"hermes-core/type-validator"}],4:[function(e,t,r){t.exports=function(){return function(e,t,r){var o;for(o in e.params)e.params[o]=e.params[o][0];r()}}},{}],5:[function(e,t,r){t.exports=function(){return function(e,t,r){e.params instanceof Array&&e.params.shift(),r()}}},{}],6:[function(e,t,r){t.exports=function(e){return function(t,r,o){"object"==typeof t.appContext?(t.appContext.routeConfig=e,o()):o(new Error("appContext is undefined"))}}},{}],7:[function(e,t,r){(function(t){(function(){YUI.add("flickr-router",(function(t,r){"use strict";const o=e("hermes-core/flog")(r),n=e("hermes-core/fletrics"),i=(e("hermes-core/holidays"),e("hermes-core/config"));var s;function a(e,r,n){var i,s=this;return n.redirect?(window.location=n.redirect,t.Promise.resolve()):(document.title=n.title,t.loaderBar.progress(),e.appContext.getView(n.view,n.params,n.layout).then((function(r){return t.loaderBar.progress(),(i=r)._params&&(i._params.keyEventScope=i.name+i._yuid),e.appContext.getKeyboardManager().setCurrentKeyEventScope(i.name+i._yuid),i.set("isRootView",!0),i.initialize()})).then((function(){var r,o;t.loaderBar.progress(),o=(o=(o=(r=t.one("html")).get("className").trim()).replace(/html-[\S]+-view/,"html-"+n.view)).replace(/[\S]+-layout/,n.layout+"-layout"),r.set("className",o),e.transactionId===s.currentTransactionId&&s.app.showView(i,null,{render:!0,callback:function(r){t.loaderBar.finish(),"popstate"!==e.src&&window.scroll(0,0)}})})).catch((function(e){throw o.error("Render had an error",{err:e}),e})))}t.FlickrRouter=function(e,t,r){return this.app=e,this.setupRoutes(t),this.auth=r,this},t.FlickrRouter.prototype={setupRoutes:function(e){var t,r,o,n,i;for(o in e.patterns)this.registerParam(o,new RegExp(e.patterns[o]));for(t=0;t<e.routes.length;t++)if((n=e.routes[t]).path instanceof Array)for(r=0;r<n.path.length;r++){for(o in i={},n)"path"!==o&&(i[o]=n[o]);i.path=n.path[r],this.registerRoute(i)}else this.registerRoute(n)},registerParam:function(e,t){this.app.param(e,t)},registerRoute:function(t){var r=this;this.route(t,[function(e,t,r){"appContext"in window?e.appContext=window.appContext:window.beaconError&&window.beaconError("[flickr-router] No appcontext.",window.location.href),r()},e("hermes-core/normalize-params-hash")(),e("hermes-core/normalize-param")("nsid_or_path_alias"),e("hermes-core/normalize-path-params")(),e("hermes-core/set-route-config")(t),function(e,o,n){return s.bounceRoute(e,o,n,t,r)},function(e,r,o){return s.checkAndKick(e,r,o,t)},function(e,t,r){Object.assign(e,i.request),r()},function(e,o,n){return s.checkAndInterstitial(e,o,n,t,r)},function(e,o,n){r._processRequest(t,e,o,n)}])},render:function(e,t,r){return a.call(this,e,t,r)},route:function(e,t){var r=[];return r.push(e.path),r=r.concat(t),this.app.route.apply(this.app,r)},_processRequest:function(e,r,o,n){var i=this;r.transactionId=t.guid(),this.currentTransactionId=r.transactionId,r.appContext.startTime=Date.now(),r.appContext.getRoute(e.module).then((function(e){var n;return i.executingRoute=e,n={id:r.id,isInsecureGod:r.isInsecureGod,cookies:r.cookies,headers:r.headers,params:r.params,url:r.url.toString(),path:r.path,query:r.query,buckets:r.buckets,lang:r.lang,geo:r.geo,probableUser:r.probableUser,UA:r.UA,hasSessionCookie:r.hasSessionCookie,body:r.body,isInRebootGroups:r.isInRebootGroups},r.isGod&&(n.isGod=r.isGod),r.routeTimingStart=Date.now(),t.Promise.all([i.auth(r,o),e.run(n,o)])})).then((function(e){if(e.length>=2){e[0];r.appContext.initialView=e[1].view}return i._renderView(e,r,o,n)})).catch((function(e){return i._throwError.call(i,e,r,o,n)}))},_renderView:function(e,r,n,s){var a,c=e.length>0?e[0]:null,u=e.length>1?e[1]:null,p={nsid:!0,ispro:!0};if(!u)throw new Error("Invalid viewConfig");if(u.params=u.params||{},c){if(c.signedIn&&c.user&&c.user.username?o.info("user is signed in",{nsid:c.user.nsid,username:c.user.username._content}):o.info("user is signed out"),c.signedIn&&c.user)for(a in c.user)void 0===p[a]&&delete c.user[a];c.isInsecureGod=r.isInsecureGod,r.appContext.auth=c}return u&&void 0!==u.params&&(u.params.isOwner=r.appContext.getViewer().signedIn&&r.appContext.getViewer().nsid===u.params.nsid),u.redirect||(u.params.UA=r.UA,u.params.isMobile=r.UA.isMobileBrowser,u.layout=u.layout||i.rendering.default_layout,u.title=t.pageTitleHelper(u.title),r.appContext.routeTiming=Date.now()-r.routeTimingStart),this.render(r,n,u)},_throwError:function(e,t,r,i){var s,a=t&&t.UA&&(t.UA.isBot||t.UA.isSharingBot),c=a?n.getBotString():"",u=this;if(!r.headersSent){if(e.is404)return s=u.executingRoute.display404Error(t,e),u.render(t,r,s);if(t&&t.appContext&&e&&e.message&&e.message.indexOf("Not enough storage is available to complete this operation")>-1)return t.appContext.mitigateClientPanda("common.IE_STORAGE_ISSUE");if(t&&t.appContext&&e&&(e.moduleLoadingError||e.message.indexOf("Loading failed: Failed to load https://combo.staticflickr.com/zz/combo?")>-1))return window.beaconError&&window.beaconError("[flickr-router] Module loading Issue:"+e.message,window.location.href,e),t.appContext.mitigateClientPanda("common.MODULES_BLOCKED");try{if(n.increment("hermes.page.failures"+c),o.error("Reboot page failure",{err:e,req:t}),e.fatal?e.timeout||e.clientTimeout?(n.increment("hermes.api.timeouts"+c),o.error("Unexpected error",{err:e,metric:"api.timeouts"+c})):"Site Auth Failed"===e.message?(n.increment("hermes.siteauth.failures"),e.type="SiteAuth",e.redirect=!0):"Session Failed"===e.message?(n.increment("hermes.sessioncookie.failures"),e.type="SiteAuth",e.redirect=!0):(n.increment("hermes.api.other"+c),o.error("Unexpected error",{err:e,metric:"api.other"+c})):(n.increment("hermes.page.failures.other"+c),o.error("Unexpected error",{err:e,metric:"page.failures.other",isBot:a}),window.beaconError&&window.beaconError("[flickr-router] Unexpected page failure:"+e.message+" UA:"+(navigator&&navigator.userAgent||"").toString(),window.location.href,e)),window.beaconError&&(!e.redirect||"SiteAuth"!==e.type)){e.panda=!0;t&&t.UA&&" unsupported:"+t.UA.isUnsupportedBrowser+" ",window.beaconError("[flickr-router] "+e.message,window.location.href,e)}}catch(t){var p=t;void 0===u.executingRoute&&(p=e),o.info({err:e});try{window.beaconError&&(e.panda=!0,window.beaconError("[flickr-router] _throwError failed: "+e.message,window.location.href,e),p!==e&&window.beaconError(p.message,window.location.href,p))}catch(e){o.info("Failed trying to beacon client error",{err:e})}}return s=u.executingRoute.display500Error(t,e),u.render(t,r,s)}o.info("server render called but headers have already been sent",{timeout:t.timeout})}},s={checkAndKick:function(e,t,r,o){r()},checkAndInterstitial:function(e,t,r,o,n){r()},bounceRoute:function(e,t,r,o,n){r()}}}),"0.0.1",{requires:["oop","page-title-helper","moment","flickr-route","localizable","url"],langBundles:["misc"]})}).call(this)}).call(this,e("_process"))},{_process:17,"hermes-core/config":1,"hermes-core/fletrics":"hermes-core/fletrics","hermes-core/flog":"hermes-core/flog","hermes-core/holidays":2,"hermes-core/normalize-param":3,"hermes-core/normalize-params-hash":4,"hermes-core/normalize-path-params":5,"hermes-core/set-route-config":6}],8:[function(e,t,r){function o(){}function n(){}function i(){}o.prototype.start=i,o.prototype.end=i,n.prototype.emit=i,n.prototype.on=i,n.prototype.increment=i,n.prototype.decrement=i,n.prototype.set=i,n.prototype.sync=i,n.prototype.timer=function(){return new o},n.prototype.use=function(e){return e(this),this},t.exports=function(){return new n},t.exports.Timer=o},{}],9:[function(e,t,r){function o(e,t,r){this.topic=e,this.props=t||{},this.stack=r||[]}function n(e){return function(t,r){var o,n=Object.assign({},this.props,{lvl:e,time:new Date,topic:this.topic});"string"!=typeof t&&"number"!=typeof t||(n.msg=t),"object"==typeof t&&Object.assign(n,t),"object"==typeof r&&Object.assign(n,r);try{for(o=0;o<this.stack.length;o++)if(!1===this.stack[o].call(this,n,e))return;this.write(n,e)}catch(e){}}}(r=t.exports=function(e,t){return new o(e,t)}).LOG=10,r.INFO=20,r.WARN=30,r.ERROR=40,o.prototype=Object.create(r),o.prototype.log=n(r.LOG),o.prototype.info=n(r.INFO),o.prototype.warn=n(r.WARN),o.prototype.error=n(r.ERROR),o.prototype.write=e("./lib/server"),o.prototype.use=function(e){return this.stack.push(e),this},o.prototype.createLogger=function(e,t){return new o(e,Object.assign({},this.props,t),this.stack.concat())}},{"./lib/server":10}],10:[function(e,t,r){(r=t.exports=function(e,t){r.levels[t]in console?console[r.levels[t]](e):console.log(e)}).levels={10:"log",20:"info",30:"warn",40:"error"}},{}],11:[function(e,t,r){t.exports=function(e){var t=Object.prototype.toString.call(e);switch(t){case"[object Number]":return function(t){return t.lvl>=e};case"[object String]":return function(t){return t.topic===e};case"[object Array]":return function(t){return!!~e.indexOf(t.topic)};case"[object RegExp]":return function(t){return e.test(t.topic)};case"[object Function]":return function(t){return!!e.call(null,t)};case"[object Boolean]":return function(){return!!e};default:throw new Error("Unsupported filter type "+t+": "+e)}}},{}],12:[function(e,t,r){(function(e){(function(){var r;r="undefined"!=typeof window?window:void 0!==e?e:"undefined"!=typeof self?self:{},t.exports=r}).call(this)}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],13:[function(e,t,r){t.exports=function(e){if(!e)return!1;var t=o.call(e);return"[object Function]"===t||"function"==typeof e&&"[object RegExp]"!==t||"undefined"!=typeof window&&(e===window.setTimeout||e===window.alert||e===window.confirm||e===window.prompt)};var o=Object.prototype.toString},{}],14:[function(e,t,r){var o=e("merge");(t.exports=function(e){this.top=e.top,this.left=e.left,this.width=e.width,this.spacing=e.spacing,this.targetRowHeight=e.targetRowHeight,this.targetRowHeightTolerance=e.targetRowHeightTolerance,this.minAspectRatio=this.width/e.targetRowHeight*(1-e.targetRowHeightTolerance),this.maxAspectRatio=this.width/e.targetRowHeight*(1+e.targetRowHeightTolerance),this.edgeCaseMinRowHeight=e.edgeCaseMinRowHeight,this.edgeCaseMaxRowHeight=e.edgeCaseMaxRowHeight,this.widowLayoutStyle=e.widowLayoutStyle,this.isBreakoutRow=e.isBreakoutRow,this.items=[],this.height=0}).prototype={addItem:function(e){var t,r,n,i=this.items.concat(e),s=this.width-(i.length-1)*this.spacing,a=i.reduce((function(e,t){return e+t.aspectRatio}),0),c=s/this.targetRowHeight;return this.isBreakoutRow&&0===this.items.length&&e.aspectRatio>=1?(this.items.push(e),this.completeLayout(s/e.aspectRatio,"justify"),!0):a<this.minAspectRatio?(this.items.push(o(e)),!0):a>this.maxAspectRatio?0===this.items.length?(this.items.push(o(e)),this.completeLayout(s/a,"justify"),!0):(t=this.width-(this.items.length-1)*this.spacing,r=this.items.reduce((function(e,t){return e+t.aspectRatio}),0),n=t/this.targetRowHeight,Math.abs(a-c)>Math.abs(r-n)?(this.completeLayout(t/r,"justify"),!1):(this.items.push(o(e)),this.completeLayout(s/a,"justify"),!0)):(this.items.push(o(e)),this.completeLayout(s/a,"justify"),!0)},isLayoutComplete:function(){return this.height>0},completeLayout:function(e,t){var r,o,n,i,s,a=this.left,c=this.width-(this.items.length-1)*this.spacing;(void 0===t||["justify","center","left"].indexOf(t)<0)&&(t="left"),e!==(o=Math.max(this.edgeCaseMinRowHeight,Math.min(e,this.edgeCaseMaxRowHeight)))?(this.height=o,r=c/o/(c/e)):(this.height=e,r=1),this.items.forEach((function(e){e.top=this.top,e.width=e.aspectRatio*this.height*r,e.height=this.height,e.left=a,a+=e.width+this.spacing}),this),"justify"===t?(a-=this.spacing+this.left,n=(a-this.width)/this.items.length,i=this.items.map((function(e,t){return Math.round((t+1)*n)})),1===this.items.length?this.items[0].width-=Math.round(n):this.items.forEach((function(e,t){t>0?(e.left-=i[t-1],e.width-=i[t]-i[t-1]):e.width-=i[t]}))):"center"===t&&(s=(this.width-a)/2,this.items.forEach((function(e){e.left+=s+this.spacing}),this))},forceComplete:function(e,t){"number"==typeof t?this.completeLayout(t,this.widowLayoutStyle):this.completeLayout(this.targetRowHeight,this.widowLayoutStyle)},getItems:function(){return this.items}}},{merge:15}],15:[function(e,t,r){!function(e){var r=function(e){return n(!0===e,!1,arguments)};function o(e,t){if("object"!==i(e))return t;for(var r in t)"object"===i(e[r])&&"object"===i(t[r])?e[r]=o(e[r],t[r]):e[r]=t[r];return e}function n(e,t,n){var s=n[0],a=n.length;(e||"object"!==i(s))&&(s={});for(var c=0;c<a;++c){var u=n[c];if("object"===i(u))for(var p in u)if("__proto__"!==p){var l=e?r.clone(u[p]):u[p];s[p]=t?o(s[p],l):l}}return s}function i(e){return{}.toString.call(e).slice(8,-1).toLowerCase()}r.recursive=function(e){return n(!0===e,!0,arguments)},r.clone=function(e){var t,o,n=e,s=i(e);if("array"===s)for(n=[],o=e.length,t=0;t<o;++t)n[t]=r.clone(e[t]);else if("object"===s)for(t in n={},e)n[t]=r.clone(e[t]);return n},e?t.exports=r:window.merge=r}("object"==typeof t&&t&&"object"==typeof t.exports&&t.exports)},{}],16:[function(e,t,r){var o=function(e){return e.replace(/^\s+|\s+$/g,"")};t.exports=function(e){if(!e)return{};for(var t,r={},n=o(e).split("\n"),i=0;i<n.length;i++){var s=n[i],a=s.indexOf(":"),c=o(s.slice(0,a)).toLowerCase(),u=o(s.slice(a+1));void 0===r[c]?r[c]=u:(t=r[c],"[object Array]"===Object.prototype.toString.call(t)?r[c].push(u):r[c]=[r[c],u])}return r}},{}],17:[function(e,t,r){var o,n,i=t.exports={};function s(){throw new Error("setTimeout has not been defined")}function a(){throw new Error("clearTimeout has not been defined")}function c(e){if(o===setTimeout)return setTimeout(e,0);if((o===s||!o)&&setTimeout)return o=setTimeout,setTimeout(e,0);try{return o(e,0)}catch(t){try{return o.call(null,e,0)}catch(t){return o.call(this,e,0)}}}!function(){try{o="function"==typeof setTimeout?setTimeout:s}catch(e){o=s}try{n="function"==typeof clearTimeout?clearTimeout:a}catch(e){n=a}}();var u,p=[],l=!1,h=-1;function f(){l&&u&&(l=!1,u.length?p=u.concat(p):h=-1,p.length&&d())}function d(){if(!l){var e=c(f);l=!0;for(var t=p.length;t;){for(u=p,p=[];++h<t;)u&&u[h].run();h=-1,t=p.length}u=null,l=!1,function(e){if(n===clearTimeout)return clearTimeout(e);if((n===a||!n)&&clearTimeout)return n=clearTimeout,clearTimeout(e);try{n(e)}catch(t){try{return n.call(null,e)}catch(t){return n.call(this,e)}}}(e)}}function g(e,t){this.fun=e,this.array=t}function m(){}i.nextTick=function(e){var t=new Array(arguments.length-1);if(arguments.length>1)for(var r=1;r<arguments.length;r++)t[r-1]=arguments[r];p.push(new g(e,t)),1!==p.length||l||c(d)},g.prototype.run=function(){this.fun.apply(null,this.array)},i.title="browser",i.browser=!0,i.env={},i.argv=[],i.version="",i.versions={},i.on=m,i.addListener=m,i.once=m,i.off=m,i.removeListener=m,i.removeAllListeners=m,i.emit=m,i.prependListener=m,i.prependOnceListener=m,i.listeners=function(e){return[]},i.binding=function(e){throw new Error("process.binding is not supported")},i.cwd=function(){return"/"},i.chdir=function(e){throw new Error("process.chdir is not supported")},i.umask=function(){return 0}},{}],18:[function(e,t,r){"use strict";var o=Object.prototype.hasOwnProperty;function n(e){try{return decodeURIComponent(e.replace(/\+/g," "))}catch(e){return null}}function i(e){try{return encodeURIComponent(e)}catch(e){return null}}r.stringify=function(e,t){t=t||"";var r,n,s=[];for(n in"string"!=typeof t&&(t="?"),e)if(o.call(e,n)){if((r=e[n])||null!=r&&!isNaN(r)||(r=""),n=i(n),r=i(r),null===n||null===r)continue;s.push(n+"="+r)}return s.length?t+s.join("&"):""},r.parse=function(e){for(var t,r=/([^=?#&]+)=?([^&]*)/g,o={};t=r.exec(e);){var i=n(t[1]),s=n(t[2]);null===i||null===s||i in o||(o[i]=s)}return o}},{}],19:[function(e,t,r){"use strict";t.exports=function(e,t){if(t=t.split(":")[0],!(e=+e))return!1;switch(t){case"http":case"ws":return 80!==e;case"https":case"wss":return 443!==e;case"ftp":return 21!==e;case"gopher":return 70!==e;case"file":return!1}return 0!==e}},{}],20:[function(e,t,r){function o(e){var t=!1;return function(){if(!t)return t=!0,e.apply(this,arguments)}}t.exports=o,o.proto=o((function(){Object.defineProperty(Function.prototype,"once",{value:function(){return o(this)},configurable:!0})}))},{}],21:[function(e,t,r){t.exports=function(){for(var e={},t=0;t<arguments.length;t++){var r=arguments[t];for(var n in r)o.call(r,n)&&(e[n]=r[n])}return e};var o=Object.prototype.hasOwnProperty},{}],"hermes-core/fletrics":[function(e,t,r){var o=e("@flickr/fletrics"),n=o();n.getBotString=function(){return".bot"},n.createStopwatch=function(e){return this.timer("hermes."+e)},o.Timer.prototype.stop=function(){return this.end()},t.exports=n},{"@flickr/fletrics":8}],"hermes-core/flog":[function(e,t,r){var o=e("@flickr/flog")("hermes"),n=e("@flickr/flog/lib/plugins/filter");o.use(n(YUI_config.flickr.log_level.browser)),o.use((function(e){e.err&&e.err instanceof Error&&(e.msg+="\n\nmessage:\n"+e.err.message,e.msg+="\n\nstack:\n"+e.err.stack)})),o.use((function(e){var t="["+e.topic+"] "+e.msg,r=o.write.levels[e.lvl];return delete e.msg,delete e.topic,delete e.lvl,r in console?console[r](t,e):console.log(t,e),!1})),t.exports=function(e){return o.createLogger(e)}},{"@flickr/flog":9,"@flickr/flog/lib/plugins/filter":11}],"hermes-core/type-validator":[function(e,t,r){function o(e){return function(t){return e.test(t)}}r.nsid=o(/^[0-9]+@N[0-9]+$/),r.pathAlias=o(/^[0-9a-zA-Z-_]+$/),r.photoId=o(/^[0-9]+$/),r.bookId=o(/^[0-9]+$/),r.orderId=o(/^[0-9]+$/)},{}],"html-truncate":[function(e,t,r){t.exports=function(e,t,r){var o,n,i,s,a,c=10>t?t:10,u=["img"],p=[],l=0,h="",f='([\\w|-]+\\s*=\\s*"[^"]*"\\s*)*',d=new RegExp("<\\/?\\w+\\s*"+f+"\\s*\\/\\s*>"),g=new RegExp("<\\/?\\w+\\s*"+f+"\\s*\\/?\\s*>"),m=/(((ftp|https?):\/\/)[\-\w@:%_\+.~#?,&\/\/=]+)|((mailto:)?[_.\w\-]+@([\w][\w\-]+\.)+[a-zA-Z]{2,3})/g,w=new RegExp("<img\\s*"+f+"\\s*\\/?\\s*>"),y=new RegExp("\\W+","g"),v=!0;function b(e){var t=e.indexOf(" ");if(-1===t&&-1===(t=e.indexOf(">")))throw new Error("HTML tag is not well-formed : "+e);return e.substring(1,t)}function _(e,o){var n,i,s=t-l,a=s,c=s<r.slop,u=c?s:r.slop-1,p=c?0:s-r.slop,h=o||s+r.slop;if(!r.truncateLastWord){if(n=e.slice(p,h),o&&n.length<=o)a=n.length;else for(;null!==(i=y.exec(n));){if(!(i.index<u)){if(i.index===u){a=s;break}a=s+(i.index-u);break}if(a=s-(u-i.index),0===i.index&&s<=1)break}e.charAt(a-1).match(/\s$/)&&a--}return a}for((r=r||{}).ellipsis=void 0!==r.ellipsis?r.ellipsis:"...",r.truncateLastWord=void 0===r.truncateLastWord||r.truncateLastWord,r.slop=void 0!==r.slop?r.slop:c;v;){if(!(v=g.exec(e))){if(l>=t)break;if(!(v=m.exec(e))||v.index>=t){h+=e.substring(0,_(e));break}for(;v;)o=v[0],n=v.index,h+=e.substring(0,n+o.length-l),e=e.substring(n+o.length),v=m.exec(e);break}if(o=v[0],n=v.index,l+n>t){h+=e.substring(0,_(e,n));break}l+=n,h+=e.substring(0,n),"/"===o[1]?(p.pop(),s=null):(s=d.exec(o))||(i=b(o),p.push(i)),h+=s?s[0]:o,e=e.substring(n+o.length)}return e.length>t-l&&r.ellipsis&&(h+=r.ellipsis),h+=(a="",p.reverse().forEach((function(e,t){-1===u.indexOf(e)&&(a+="</"+e+">")})),a),r.keepImageTag||(h=function(e){var t,r,o=w.exec(e);return o?(t=o.index,r=o[0].length,e.substring(0,t)+e.substring(t+r)):e}(h)),h}},{}],int:[function(e,t,r){var o=function(e){if(!(this instanceof o))return new o(e);var t=this;if(e instanceof o)return t._s=e._s,void(t._d=e._d.slice());t._s="-"===(e+="").charAt(0)?1:0,t._d=[];for(var r=(e=e.replace(/[^\d]/g,"")).length,n=0;n<r;++n)t._d.push(+e[n]);i(t),0===t._d.length&&(t._s=0)};o.prototype.add=function(e){var t=this;e=n(e);if(t._s!=e._s){e._s^=1;var r=t.sub(e);return e._s^=1,r}if(t._d.length<e._d.length)var i=t._d,s=e._d,a=o(e);else i=e._d,s=t._d,a=o(t);for(var c=i.length,u=s.length,p=(r=a._d,0),l=u-1,h=c-1;h>=0;--l,--h)r[l]+=p+i[h],p=0,r[l]>=10&&(r[l]-=10,p=1);for(;l>=0&&(r[l]+=p,p=0,r[l]>=10&&(r[l]-=10,p=1),0!==p);--l);return p>0&&r.unshift(1),a},o.prototype.sub=function(e){var t=this;e=o(e);if(t._s!=e._s){e._s^=1;var r=this.add(e);return e._s^=1,r}var n=t._s,s=e._s;t._s=e._s=0;var a=t.lt(e),c=a?t._d:e._d,u=a?e._d:t._d;t._s=n,e._s=s;var p=c.length,l=u.length,h=o(a?e:t);h._s=e._s&t._s;r=h._d;for(var f=0,d=l-1,g=p-1;g>=0;--d,--g)r[d]-=c[g]+f,f=0,r[d]<0&&(r[d]+=10,f=1);for(;d>=0&&(r[d]-=f,f=0,r[d]<0&&(r[d]+=10,f=1),0!==f);--d);return a&&(h._s^=1),i(h),0===h._d.length&&(h._s=0),h},o.prototype.mul=function(e){for(var t=this,r=t._d.length>=(e=o(e))._d.length,n=(r?t:e)._d,i=(r?e:t)._d,s=n.length,a=i.length,c=o(),u=[],p=a-1;p>=0;--p){for(var l=o(),h=l._d=l._d.concat(u),f=0,d=s-1;d>=0;--d){var g=i[p]*n[d]+f,m=g%10;f=Math.floor(g/10),h.unshift(m)}f&&h.unshift(f),c=c.add(l),u.push(0)}return c._s=t._s^e._s,c},o.prototype.div=function(e){var t=this;if("0"==(e=o(e)))throw new Error("Division by 0");if("0"==t)return o();var r=t._d.slice(),n=o();n._s=t._s^e._s;var s=e._s;e._s=0;for(var a=o();r.length;){for(var c=0;r.length&&a.lt(e);)c++>0&&n._d.push(0),a._d.push(r.shift()),i(a);for(var u=0;a.gte(e)&&++u;)a=a.sub(e);if(0===u){n._d.push(0);break}n._d.push(u)}var p=a._d.length;return(p>1||n._s&&p>0)&&(a=a.add(5)),n._s&&(p!==a._d.length||a._d[0]>=5)&&(n=n.sub(1)),e._s=s,i(n)},o.prototype.mod=function(e){return this.sub(this.div(e).mul(e))},o.prototype.pow=function(e){var t=o(this);if(0==(e=o(e)))return t.set(1);for(var r=Math.abs(e);--r;t.set(t.mul(this)));return e<0?t.set(o(1).div(t)):t},o.prototype.set=function(e){return this.constructor(e),this},o.prototype.cmp=function(e){var t=this;e=n(e);if(t._s!=e._s)return t._s?-1:1;var r=t._d,o=e._d,i=r.length,s=o.length;if(i!=s)return i>s^t._s?1:-1;for(var a=0;a<i;++a)if(r[a]!=o[a])return r[a]>o[a]^t._s?1:-1;return 0},o.prototype.neg=function(){var e=o(this);return e._s^=1,e},o.prototype.abs=function(){var e=o(this);return e._s=0,e};function n(e){return e instanceof o?e:o(e)}function i(e){for(;e._d.length&&0===e._d[0];)e._d.shift();return e}o.prototype.valueOf=o.prototype.toString=function(e){var t=this;if(!e||10===e)return(t._s&&t._d.length?"-":"")+(t._d.length?t._d.join(""):"0");if(e<2||e>36)throw RangeError("radix out of range: "+e);for(var r=Math.pow(e,6),o=t,n="";;){var i=o.div(r),s=(+o.sub(i.mul(r)).toString()).toString(e);if((o=i).eq(0))return s+n;for(;s.length<6;)s="0"+s;n=""+s+n}},o.prototype.gt=function(e){return this.cmp(e)>0},o.prototype.gte=function(e){return this.cmp(e)>=0},o.prototype.eq=function(e){return 0===this.cmp(e)},o.prototype.ne=function(e){return 0!==this.cmp(e)},o.prototype.lt=function(e){return this.cmp(e)<0},o.prototype.lte=function(e){return this.cmp(e)<=0},t.exports=o},{}],"justified-layout":[function(e,t,r){"use strict";var o=e("merge"),n=e("./row");function i(e,t){var r;return!1!==e.fullWidthBreakoutRowCadence&&(t._rows.length+1)%e.fullWidthBreakoutRowCadence==0&&(r=!0),new n({top:t._containerHeight,left:e.containerPadding.left,width:e.containerWidth-e.containerPadding.left-e.containerPadding.right,spacing:e.boxSpacing.horizontal,targetRowHeight:e.targetRowHeight,targetRowHeightTolerance:e.targetRowHeightTolerance,edgeCaseMinRowHeight:.5*e.targetRowHeight,edgeCaseMaxRowHeight:2*e.targetRowHeight,rightToLeft:!1,isBreakoutRow:r,widowLayoutStyle:e.widowLayoutStyle})}function s(e,t,r){return t._rows.push(r),t._layoutItems=t._layoutItems.concat(r.getItems()),t._containerHeight+=r.height+e.boxSpacing.vertical,r.items}t.exports=function(e,t){var r={},n={},a={containerWidth:1060,containerPadding:10,boxSpacing:10,targetRowHeight:320,targetRowHeightTolerance:.25,maxNumRows:Number.POSITIVE_INFINITY,forceAspectRatio:!1,showWidows:!0,fullWidthBreakoutRowCadence:!1,widowLayoutStyle:"left"},c={},u={};return r=o(a,t=t||{}),c.top=isNaN(parseFloat(r.containerPadding.top))?r.containerPadding:r.containerPadding.top,c.right=isNaN(parseFloat(r.containerPadding.right))?r.containerPadding:r.containerPadding.right,c.bottom=isNaN(parseFloat(r.containerPadding.bottom))?r.containerPadding:r.containerPadding.bottom,c.left=isNaN(parseFloat(r.containerPadding.left))?r.containerPadding:r.containerPadding.left,u.horizontal=isNaN(parseFloat(r.boxSpacing.horizontal))?r.boxSpacing:r.boxSpacing.horizontal,u.vertical=isNaN(parseFloat(r.boxSpacing.vertical))?r.boxSpacing:r.boxSpacing.vertical,r.containerPadding=c,r.boxSpacing=u,n._layoutItems=[],n._awakeItems=[],n._inViewportItems=[],n._leadingOrphans=[],n._trailingOrphans=[],n._containerHeight=r.containerPadding.top,n._rows=[],n._orphans=[],r._widowCount=0,function(e,t,r){var o,n,a,c=[];return e.forceAspectRatio&&r.forEach((function(t){t.forcedAspectRatio=!0,t.aspectRatio=e.forceAspectRatio})),r.some((function(r,a){if(isNaN(r.aspectRatio))throw new Error("Item "+a+" has an invalid aspect ratio");if(n||(n=i(e,t)),o=n.addItem(r),n.isLayoutComplete()){if(c=c.concat(s(e,t,n)),t._rows.length>=e.maxNumRows)return n=null,!0;if(n=i(e,t),!o&&(o=n.addItem(r),n.isLayoutComplete())){if(c=c.concat(s(e,t,n)),t._rows.length>=e.maxNumRows)return n=null,!0;n=i(e,t)}}})),n&&n.getItems().length&&e.showWidows&&(t._rows.length?(a=t._rows[t._rows.length-1].isBreakoutRow?t._rows[t._rows.length-1].targetRowHeight:t._rows[t._rows.length-1].height,n.forceComplete(!1,a)):n.forceComplete(!1),c=c.concat(s(e,t,n)),e._widowCount=n.getItems().length),t._containerHeight=t._containerHeight-e.boxSpacing.vertical,t._containerHeight=t._containerHeight+e.containerPadding.bottom,{containerHeight:t._containerHeight,widowCount:e._widowCount,boxes:t._layoutItems}}(r,n,e.map((function(e){return e.width&&e.height?{aspectRatio:e.width/e.height}:{aspectRatio:e}})))}},{"./row":14,merge:15}],"url-parse":[function(e,t,r){(function(r){(function(){"use strict";var o=e("requires-port"),n=e("querystringify"),i=/^[\x00-\x20\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+/,s=/[\n\r\t]/g,a=/^[A-Za-z][A-Za-z0-9+-.]*:\/\//,c=/:\d+$/,u=/^([a-z][a-z0-9.+-]*:)?(\/\/)?([\\/]+)?([\S\s]*)/i,p=/^[a-zA-Z]:/;function l(e){return(e||"").toString().replace(i,"")}var h=[["#","hash"],["?","query"],function(e,t){return g(t.protocol)?e.replace(/\\/g,"/"):e},["/","pathname"],["@","auth",1],[NaN,"host",void 0,1,1],[/:(\d*)$/,"port",void 0,1],[NaN,"hostname",void 0,1,1]],f={hash:1,query:1};function d(e){var t,o=("undefined"!=typeof window?window:void 0!==r?r:"undefined"!=typeof self?self:{}).location||{},n={},i=typeof(e=e||o);if("blob:"===e.protocol)n=new w(unescape(e.pathname),{});else if("string"===i)for(t in n=new w(e,{}),f)delete n[t];else if("object"===i){for(t in e)t in f||(n[t]=e[t]);void 0===n.slashes&&(n.slashes=a.test(e.href))}return n}function g(e){return"file:"===e||"ftp:"===e||"http:"===e||"https:"===e||"ws:"===e||"wss:"===e}function m(e,t){e=(e=l(e)).replace(s,""),t=t||{};var r,o=u.exec(e),n=o[1]?o[1].toLowerCase():"",i=!!o[2],a=!!o[3],c=0;return i?a?(r=o[2]+o[3]+o[4],c=o[2].length+o[3].length):(r=o[2]+o[4],c=o[2].length):a?(r=o[3]+o[4],c=o[3].length):r=o[4],"file:"===n?c>=2&&(r=r.slice(2)):g(n)?r=o[4]:n?i&&(r=r.slice(2)):c>=2&&g(t.protocol)&&(r=o[4]),{protocol:n,slashes:i||g(n),slashesCount:c,rest:r}}function w(e,t,r){if(e=(e=l(e)).replace(s,""),!(this instanceof w))return new w(e,t,r);var i,a,c,u,f,y,v=h.slice(),b=typeof t,_=this,x=0;for("object"!==b&&"string"!==b&&(r=t,t=null),r&&"function"!=typeof r&&(r=n.parse),i=!(a=m(e||"",t=d(t))).protocol&&!a.slashes,_.slashes=a.slashes||i&&t.slashes,_.protocol=a.protocol||t.protocol||"",e=a.rest,("file:"===a.protocol&&(2!==a.slashesCount||p.test(e))||!a.slashes&&(a.protocol||a.slashesCount<2||!g(_.protocol)))&&(v[3]=[/(.*)/,"pathname"]);x<v.length;x++)"function"!=typeof(u=v[x])?(c=u[0],y=u[1],c!=c?_[y]=e:"string"==typeof c?~(f="@"===c?e.lastIndexOf(c):e.indexOf(c))&&("number"==typeof u[2]?(_[y]=e.slice(0,f),e=e.slice(f+u[2])):(_[y]=e.slice(f),e=e.slice(0,f))):(f=c.exec(e))&&(_[y]=f[1],e=e.slice(0,f.index)),_[y]=_[y]||i&&u[3]&&t[y]||"",u[4]&&(_[y]=_[y].toLowerCase())):e=u(e,_);r&&(_.query=r(_.query)),i&&t.slashes&&"/"!==_.pathname.charAt(0)&&(""!==_.pathname||""!==t.pathname)&&(_.pathname=function(e,t){if(""===e)return t;for(var r=(t||"/").split("/").slice(0,-1).concat(e.split("/")),o=r.length,n=r[o-1],i=!1,s=0;o--;)"."===r[o]?r.splice(o,1):".."===r[o]?(r.splice(o,1),s++):s&&(0===o&&(i=!0),r.splice(o,1),s--);return i&&r.unshift(""),"."!==n&&".."!==n||r.push(""),r.join("/")}(_.pathname,t.pathname)),"/"!==_.pathname.charAt(0)&&g(_.protocol)&&(_.pathname="/"+_.pathname),o(_.port,_.protocol)||(_.host=_.hostname,_.port=""),_.username=_.password="",_.auth&&(~(f=_.auth.indexOf(":"))?(_.username=_.auth.slice(0,f),_.username=encodeURIComponent(decodeURIComponent(_.username)),_.password=_.auth.slice(f+1),_.password=encodeURIComponent(decodeURIComponent(_.password))):_.username=encodeURIComponent(decodeURIComponent(_.auth)),_.auth=_.password?_.username+":"+_.password:_.username),_.origin="file:"!==_.protocol&&g(_.protocol)&&_.host?_.protocol+"//"+_.host:"null",_.href=_.toString()}w.prototype={set:function(e,t,r){var i=this;switch(e){case"query":"string"==typeof t&&t.length&&(t=(r||n.parse)(t)),i[e]=t;break;case"port":i[e]=t,o(t,i.protocol)?t&&(i.host=i.hostname+":"+t):(i.host=i.hostname,i[e]="");break;case"hostname":i[e]=t,i.port&&(t+=":"+i.port),i.host=t;break;case"host":i[e]=t,c.test(t)?(t=t.split(":"),i.port=t.pop(),i.hostname=t.join(":")):(i.hostname=t,i.port="");break;case"protocol":i.protocol=t.toLowerCase(),i.slashes=!r;break;case"pathname":case"hash":if(t){var s="pathname"===e?"/":"#";i[e]=t.charAt(0)!==s?s+t:t}else i[e]=t;break;case"username":case"password":i[e]=encodeURIComponent(t);break;case"auth":var a=t.indexOf(":");~a?(i.username=t.slice(0,a),i.username=encodeURIComponent(decodeURIComponent(i.username)),i.password=t.slice(a+1),i.password=encodeURIComponent(decodeURIComponent(i.password))):i.username=encodeURIComponent(decodeURIComponent(t))}for(var u=0;u<h.length;u++){var p=h[u];p[4]&&(i[p[1]]=i[p[1]].toLowerCase())}return i.auth=i.password?i.username+":"+i.password:i.username,i.origin="file:"!==i.protocol&&g(i.protocol)&&i.host?i.protocol+"//"+i.host:"null",i.href=i.toString(),i},toString:function(e){e&&"function"==typeof e||(e=n.stringify);var t,r=this,o=r.host,i=r.protocol;i&&":"!==i.charAt(i.length-1)&&(i+=":");var s=i+(r.protocol&&r.slashes||g(r.protocol)?"//":"");return r.username?(s+=r.username,r.password&&(s+=":"+r.password),s+="@"):r.password?(s+=":"+r.password,s+="@"):"file:"!==r.protocol&&g(r.protocol)&&!o&&"/"!==r.pathname&&(s+="@"),(":"===o[o.length-1]||c.test(r.hostname)&&!r.port)&&(o+=":"),s+=o+r.pathname,(t="object"==typeof r.query?e(r.query):r.query)&&(s+="?"!==t.charAt(0)?"?"+t:t),r.hash&&(s+=r.hash),s}},w.extractProtocol=m,w.location=d,w.trimLeft=l,w.qs=n,t.exports=w}).call(this)}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{querystringify:18,"requires-port":19}],xhr:[function(e,t,r){"use strict";var o=e("global/window"),n=e("once"),i=e("is-function"),s=e("parse-headers"),a=e("xtend");function c(e,t,r){var o=e;return i(t)?(r=t,"string"==typeof e&&(o={uri:e})):o=a(t,{uri:e}),o.callback=r,o}function u(e,t,r){return p(t=c(e,t,r))}function p(e){var t=e.callback;if(void 0===t)throw new Error("callback argument missing");function r(){var e=void 0;if(l.response?e=l.response:"text"!==l.responseType&&l.responseType||(e=l.responseText||l.responseXML),y)try{e=JSON.parse(e)}catch(e){}return e}t=n(t);var o={body:void 0,headers:{},statusCode:0,method:d,url:f,rawRequest:l};function i(e){clearTimeout(h),e instanceof Error||(e=new Error(""+(e||"Unknown XMLHttpRequest Error"))),e.statusCode=0,t(e,o)}function a(){if(!p){var n;clearTimeout(h),n=e.useXDR&&void 0===l.status?200:1223===l.status?204:l.status;var i=o,a=null;0!==n?(i={body:r(),statusCode:n,method:d,headers:{},url:f,rawRequest:l},l.getAllResponseHeaders&&(i.headers=s(l.getAllResponseHeaders()))):a=new Error("Internal XMLHttpRequest Error"),t(a,i,i.body)}}var c,p,l=e.xhr||null;l||(l=e.cors||e.useXDR?new u.XDomainRequest:new u.XMLHttpRequest);var h,f=l.url=e.uri||e.url,d=l.method=e.method||"GET",g=e.body||e.data||null,m=l.headers=e.headers||{},w=!!e.sync,y=!1;if("json"in e&&(y=!0,m.accept||m.Accept||(m.Accept="application/json"),"GET"!==d&&"HEAD"!==d&&(m["content-type"]||m["Content-Type"]||(m["Content-Type"]="application/json"),g=JSON.stringify(e.json))),l.onreadystatechange=function(){4===l.readyState&&a()},l.onload=a,l.onerror=i,l.onprogress=function(){},l.ontimeout=i,l.open(d,f,!w,e.username,e.password),w||(l.withCredentials=!!e.withCredentials),!w&&e.timeout>0&&(h=setTimeout((function(){p=!0,l.abort("timeout");var e=new Error("XMLHttpRequest timeout");e.code="ETIMEDOUT",i(e)}),e.timeout)),l.setRequestHeader)for(c in m)m.hasOwnProperty(c)&&l.setRequestHeader(c,m[c]);else if(e.headers&&!function(e){for(var t in e)if(e.hasOwnProperty(t))return!1;return!0}(e.headers))throw new Error("Headers cannot be set on an XDomainRequest object");return"responseType"in e&&(l.responseType=e.responseType),"beforeSend"in e&&"function"==typeof e.beforeSend&&e.beforeSend(l),l.send(g),l}t.exports=u,u.XMLHttpRequest=o.XMLHttpRequest||function(){},u.XDomainRequest="withCredentials"in new u.XMLHttpRequest?u.XMLHttpRequest:o.XDomainRequest,function(e,t){for(var r=0;r<e.length;r++)t(e[r])}(["get","put","post","patch","head","delete"],(function(e){u["delete"===e?"del":e]=function(t,r,o){return(r=c(t,r,o)).method=e.toUpperCase(),p(r)}}))},{"global/window":12,"is-function":13,once:20,"parse-headers":16,xtend:21}]},{},[7]);